//! Concrete backend storage types and their host-side capabilities.
//!
//! The layer below [`crate::tensor`]: each [`crate::backend::Backend`] picks one of these as its
//! `Storage` (Simulation → [`MathStorage`], Emulation/Npu → [`BufStorage`], Typecheck →
//! [`PhantomStorage`]). The `Tensor` wrapper drives them through the `Backend` seam; the storage
//! types own their buffers and implement the lifecycle constructors/serializers and value-ops as
//! inherent methods.

use abi_stable::std_types::RResult;
use furiosa_mapping::*;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

mod buf;
mod math;
mod par_iters;
mod phantom;

pub use buf::BufStorage;
pub use math::MathStorage;
pub use phantom::PhantomStorage;

/// A raw `*mut T` + length carried into a rayon closure for a disjoint-index scatter. `Sync` is the
/// unchecked part: the caller must guarantee every parallel write targets a *distinct* index (no two
/// jobs touch the same slot), which is what makes the concurrent writes through this pointer
/// data-race-free. The `len` bound is checked, so an out-of-range index is a caught panic, not UB.
/// The pointer is private and [`Self::write`] is the sole write entry point, so a caller cannot read
/// it back or write through the bare pointer by accident. Drive it through [`par_scatter`] rather than
/// hand-rolling the rayon loop where the plan is one write per step.
pub(crate) struct DisjointScatter<T> {
    base: *mut T,
    len: usize,
}
// SAFETY: see the type doc -- every parallel write hits a distinct slot, so sharing the pointer is sound.
unsafe impl<T> Sync for DisjointScatter<T> {}

impl<T> DisjointScatter<T> {
    /// Wraps a destination slice's base pointer and length.
    pub(crate) fn new(dst: &mut [T]) -> Self {
        Self {
            base: dst.as_mut_ptr(),
            len: dst.len(),
        }
    }

    /// Writes `value` at element `idx`, bounds-checking `idx` against the wrapped length (a violation
    /// is a caught panic in every build, not UB).
    ///
    /// # Safety
    /// No two concurrent calls (across rayon jobs) may share an `idx` -- the caller's disjoint-write
    /// invariant (see the type doc). The bound is checked here; distinctness is the caller's.
    pub(crate) unsafe fn write(&self, idx: usize, value: T) {
        assert!(idx < self.len, "DisjointScatter index {idx} out of bounds {}", self.len);
        // SAFETY: `idx < self.len` (asserted) bounds the offset; distinct-index is the caller's invariant.
        unsafe { *self.base.add(idx) = value };
    }
}

/// Scatters `plan`'s writes into `dst` across the rayon pool. Consolidates the disjoint-write
/// parallel-scatter loop shared by the storages' relayouts (transpose / reshape / scatter_buf /
/// canonical_bases): each produces one `Option<(index, value)>` per plan step in a safe closure and
/// hands them here, so the sole raw write and its bound live in one place ([`DisjointScatter::write`]).
/// Each step is `Some((idx, value))` to write or `None` to skip -- the `Option` (rather than a
/// `filter`) keeps the plan an [`IndexedParallelIterator`], so `with_min_len` can hold a small scatter
/// to one (serial) job. Sites that SKIP an out-of-range or padding write yield `None`; a site where an
/// out-of-range index is a bug yields `Some` unconditionally and lets the write's bound assert fail loud.
///
/// # Safety invariant (caller)
/// The `Some` indices must be pairwise distinct (a bijection over the live cells) -- no two steps may
/// share an index, so no two jobs write one slot. The bound is checked by [`DisjointScatter::write`].
pub(crate) fn par_scatter<T: Send>(dst: &mut [T], plan: impl IndexedParallelIterator<Item = Option<(usize, T)>>) {
    let scatter = DisjointScatter::new(dst);
    let scatter = &scatter;
    plan.with_min_len(PAR_MIN_JOB).for_each(|write| {
        if let Some((idx, value)) = write {
            // SAFETY: the caller guarantees distinct indices (see the fn doc); `write` bounds `idx`.
            unsafe { scatter.write(idx, value) };
        }
    });
}

/// Minimum work per parallel job, in scalar operations (multiply-adds / fold steps), for the storages'
/// rayon-parallel ops (`contraction`, `reduce`, `transpose`): rayon never splits below this, so a small
/// op stays one (effectively serial) job rather than fragmenting into micro-jobs whose dispatch overhead
/// would dominate (acute on a many-core box, where rayon splits aggressively). Each call site converts
/// it to its own split unit: an op whose split element does one scalar op (a stream position) uses it as
/// `with_min_len` directly, while a per-output-cell op divides by the inner block length so a job still
/// does about `PAR_MIN_JOB` scalar ops.
pub(crate) const PAR_MIN_JOB: usize = 1 << 16;

/// The `with_min_len` for a per-output-cell parallel op: how many output cells make one job do about
/// [`PAR_MIN_JOB`] scalar ops, given each cell folds an `inner_block`-length run. Clamped to `>= 1` so a
/// huge inner block does not yield `with_min_len(0)` (rayon would then over-split). Shared by the
/// storages' per-cell `reduce` / `contraction` / `gather`.
pub(crate) fn min_cells_per_job(inner_block: usize) -> usize {
    (PAR_MIN_JOB / inner_block.max(1)).max(1)
}

// Concrete per-backend storage types.
//
// Three concrete storage types, [`MathStorage`] (Simulation: canonical `ArrayD<Opt<D>>`
// interpreter), [`BufStorage`] (Emulation / Npu: physical `Vec<D>` staging buffer) and
// [`PhantomStorage`] (Typecheck: axes only), each own their storage and implement the lifecycle
// constructors/serializers and value-ops as inherent methods. The single backend seam
// ([`crate::backend::Backend`]) delegates to these.
//
// The storage `Mapping` type parameter is fully erased: each concrete storage is `XStorage<D>` and
// the backend GAT is `Storage<D>`. Element-wise ops (`map` / `zip_with` / `zip3_with`) are
// storage-native and mapping-free (they preserve the source layout). Mapping-changing ops
// (`transpose` / `reduce` / `scatter` / `gather` / `reshape`) keep their source/target mapping
// *type* parameters, the storage no longer carries the mapping, so these take it explicitly to
// drive the relayout.
//
// ## Offset address space is backend-private
//
// Where a per-backend body addresses storage by a flat offset (e.g. `BufStorage`'s `self.data[i]`),
// that address space is NOT portable across backends: `MathStorage` offsets would be dense (product
// of the axes' moduli, no padding, innermost axis contiguous); `BufStorage` offsets are physical (`Mapping::SIZE`,
// padding included); `PhantomStorage` has none. The only contract is: two tensors of the **same
// backend and mapping** address the same logical position by the same offset. That suffices for the
// element-wise ops (same mapping) and for the per-backend restructuring bodies (which relate two
// mappings using their own backend's convention).

/// Resolves an `Index` to a multi-dim coordinate vector against `axes`. Returns `None` when the
/// index lands in a padding slot. `finalize` gives each symbol's absolute coordinate; recover this
/// axis's digit by its stride/modulo, since an axis may be a sub-factor of its symbol (`A % 4`).
///
/// Shared by [`MathStorage`]'s index decode and the vector engine's `AxisToggle` host pattern.
pub(crate) fn axis_coords(axes: &[AxisTerm], index: Index) -> Option<Vec<usize>> {
    let RResult::ROk(coords) = index.finalize() else {
        return None;
    };
    Some(
        axes.iter()
            .map(|axis| (coords.get(&axis.symbol).copied().unwrap_or(0) / axis.stride) % axis.modulo)
            .collect(),
    )
}

/// The position of the live axis term whose symbol is `axis` within `axes`, or `None` if absent.
/// Lets [`MathStorage`] and [`BufStorage`] map a per-cell coordinate vector to a single axis's
/// coordinate when building an axis-index tensor.
pub(crate) fn axis_position(axes: &[AxisTerm], axis: &Ident) -> Option<usize> {
    axes.iter().position(|term| term.symbol == *axis)
}

#[cfg(test)]
mod concrete_tests {
    use furiosa_mapping::*;

    use super::*;
    use crate::scalar::Opt;

    axes![A = 4, B = 2];
    axes![P = 4, Q = 4];
    axes![H = 3, I = 2, E = 2];
    axes![Hp = 3, Ep = 4];

    /// A zero-filled `BufStorage` dst canvas of `mapping`'s size (the relayout overwrites the live
    /// cells; padding stays zero), the `Backend::uninit` blank canvas for BUF, built via `from_buf`.
    fn buf_zeros(mapping: &Mapping) -> BufStorage<i32> {
        BufStorage::from_buf(mapping, std::iter::repeat_n(0, mapping.size()))
    }

    /// `MathStorage` and `BufStorage` reduce identically through the sibling-free `reduce`,
    /// each driving its own backend body (plan-walked offsets vs sequencer) but reaching the same
    /// result.
    #[test]
    fn reduce_into_math_and_buf() {
        // m![A,B] (A=4,B=2): src[a,b] = a*2+b. reduce over B (inner): dst[a] = sum_b src[a,b].
        let math = MathStorage::<i32>::from_buf(&<m![A, B]>::to_value(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let math_out: MathStorage<i32> = math.reduce::<m![A, B], m![A], _>(|a, b| a + b, 0, false);
        assert_eq!(math_out.into_buf(&<m![A]>::to_value()), vec![1, 5, 9, 13]);

        let buf = BufStorage::<i32>::from_buf(&<m![A, B]>::to_value(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let buf_out: BufStorage<i32> = buf.reduce::<m![A, B], m![A], _>(|a, b| a + b, 0, false);
        assert_eq!(buf_out.into_buf(&<m![A]>::to_value()), vec![1, 5, 9, 13]);
    }

    /// `transpose` relayouts correctly on both backends through the sibling-free generic
    /// source.
    #[test]
    fn transpose_math_and_buf() {
        // m![A,B] (A=4,B=2): t[a,b] = a*2+b -> m![B,A]: out[b*4+a] = t[a,b].
        let math = MathStorage::<i32>::from_buf(&<m![A, B]>::to_value(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let mut math_out = MathStorage::<i32>::uninit(&<m![B, A]>::to_value());
        math_out.transpose::<m![A, B], m![B, A]>(&math, &Index::new(), &Index::new(), false);
        assert_eq!(math_out.into_buf(&<m![B, A]>::to_value()), vec![0, 2, 4, 6, 1, 3, 5, 7]);

        let buf = BufStorage::<i32>::from_buf(&<m![A, B]>::to_value(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let mut buf_out = buf_zeros(&<m![B, A]>::to_value());
        buf_out.transpose::<m![A, B], m![B, A]>(
            &buf,
            &Index::new(),
            &Index::new(),
            &<m![A, B]>::to_value(),
            &<m![B, A]>::to_value(),
            false,
        );
        assert_eq!(buf_out.into_buf(&<m![B, A]>::to_value()), vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }

    /// Partial-view (non-zero offset) transpose agrees between `MathStorage` and `BufStorage`.
    ///
    /// Parent `m![P, Q]` (P=4, Q=4). A `Q`-tile of 2 yields the view mapping `m![P, Q = 2 # 4]`, the
    /// `# 4` keeps `Q`'s parent wire extent, so the sub-view keeps the parent strides, and a non-zero
    /// `src_offset` that shifts the window in `Q` (exactly what `TensorView::tile` builds via
    /// `Index::add_mapping`). Transposing that sub-view to `m![Q = 2 # 4, P]` exercises a non-zero
    /// physical base on the source side. `BufStorage` must land the same cells `MathStorage` does.
    #[test]
    fn transpose_offset_math_and_buf() {
        // --- Source-side offset: read a shifted Q-window of the parent and transpose it. ---
        // Parent m![P, Q] (P=4, Q=4) is the full src buffer; the sub-view is m![P, Q = 2 # 4]
        // (`# 4` keeps Q's parent wire extent), offset = the window shift `tile` records for tile
        // index 1 over Q. Transpose the sub-view to m![Q = 2 # 4, P]. The live result is the Q∈{1,2}
        // columns read straight from the parent.
        let parent: Vec<i32> = (0..16).collect();
        let mut src_offset = Index::new();
        src_offset.add_mapping::<m![Q]>(1);
        assert_offset_transpose_agrees::<m![P, Q], m![P, Q = 2 # 4], m![Q = 2 # 4, P]>(
            &parent,
            &src_offset,
            &Index::new(),
            &[1, 5, 9, 13, 2, 6, 10, 14],
        );

        // --- Destination-side offset: write a compact tile into a shifted P-window of the parent. ---
        // Source m![Q = 2, P] (size 8) holds the tile densely; dst sub-view m![Q = 2, P = 2 # 4] over a
        // parent dst buffer m![Q = 2, P] (size 8) at P-offset 1. The base lands the tile at the
        // shifted dst wire positions.
        let tile: Vec<i32> = (0..8).collect();
        let mut dst_offset = Index::new();
        dst_offset.add_mapping::<m![P]>(1);
        assert_offset_transpose_agrees::<m![Q = 2, P], m![Q = 2, P = 2 # 4], m![Q = 2, P = 2 # 4]>(
            &tile,
            &Index::new(),
            &dst_offset,
            // P resized to 2 in the dst view: the live cells are the P=0 column of each Q row.
            &[0, 4],
        );
    }

    /// Interleave-tile transpose, the MNIST `begin_interleaved` shape: the dst sub-view carries the
    /// interleave axis `I` as a `# 2` Top pad (an axis the source lacks), and the `{I:1}` dst offset
    /// must land on that padded slot's wire half, a position no live cell finalizes to. `BufStorage`
    /// must write the same cells `MathStorage` does (its `to_buf_opt` `Init` slots).
    #[test]
    fn transpose_interleave_padding_offset() {
        // Source m![H, E] (no I), size 6, holds the tile densely. The dst parent m![H, I, E] (size 12)
        // interleaves I; the dst sub-view m![H, 1 # 2, E] selects I=1 via the `{I:1}` offset.
        let src_data: Vec<i32> = (0..6).collect();
        let src_map = <m![H, E]>::to_value();
        let dst_map = <m![H, I, E]>::to_value();
        let mut dst_offset = Index::new();
        dst_offset.add_mapping::<m![I]>(1);

        let math = MathStorage::<i32>::from_buf(&src_map, src_data.iter().copied());
        let mut math_out = MathStorage::<i32>::uninit(&dst_map);
        math_out.transpose::<m![H, E], m![H, 1 # 2, E]>(&math, &Index::new(), &dst_offset, false);
        let math_buf = math_out.to_buf_opt(&dst_map);

        let buf = BufStorage::<i32>::from_buf(&src_map, src_data.iter().copied());
        let mut buf_out = buf_zeros(&dst_map);
        // BUF resolves the `{I:1}` dst offset against the dst *base* map (I live, m![H, I, E]); the
        // src has no offset so its base map is irrelevant.
        buf_out.transpose::<m![H, E], m![H, 1 # 2, E]>(&buf, &Index::new(), &dst_offset, &src_map, &dst_map, false);

        // The I=1 wire half must hold the source, in order; the I=0 half stays untouched.
        let mut live = Vec::new();
        for (pos, opt) in math_buf.iter().enumerate() {
            if let Opt::Init(v) = opt {
                assert_eq!(buf_out.data[pos], *v, "math/buf mismatch at dst wire position {pos}");
                live.push((pos, *v));
            }
        }
        assert_eq!(live, vec![(2, 0), (3, 1), (6, 2), (7, 3), (10, 4), (11, 5)]);
    }

    /// Interleave-tile *source* read, the MNIST `VectorTensorPair::new` shape: the source has the
    /// interleave axis `I` LIVE (`m![H, I, E]`), and a tile reads its `I = 1` slice through a sub-view
    /// that carries `I` as a `# 2` Top pad with a `{I:1}` *src* offset. The offset axis is padding in
    /// the `Src` *type* but live in the source's base map, BUF must resolve the wire base against the
    /// base map (where `I` is a live `Symbol`), reading the same cells `MathStorage` does.
    #[test]
    fn transpose_interleave_padding_src_offset() {
        // Source parent m![H, I, E] (size 12), I live. Read its I=1 slice into a dst sub-view that
        // carries I as `1 # 2` padding (size 12), dst offset empty.
        let data: Vec<i32> = (0..12).collect();
        let src_parent = <m![H, I, E]>::to_value();
        let dst_view = <m![H, 1 # 2, E]>::to_value();
        let mut src_offset = Index::new();
        src_offset.add_mapping::<m![I]>(1);

        let math_src = MathStorage::<i32>::from_buf(&src_parent, data.iter().copied());
        let mut math_dst = MathStorage::<i32>::uninit(&dst_view);
        math_dst.transpose::<m![H, 1 # 2, E], m![H, 1 # 2, E]>(&math_src, &src_offset, &Index::new(), false);
        let math_buf = math_dst.to_buf_opt(&dst_view);

        let buf_src = BufStorage::<i32>::from_buf(&src_parent, data.iter().copied());
        let mut buf_dst = buf_zeros(&dst_view);
        buf_dst.transpose::<m![H, 1 # 2, E], m![H, 1 # 2, E]>(
            &buf_src,
            &src_offset,
            &Index::new(),
            &src_parent,
            &dst_view,
            false,
        );

        let mut live = Vec::new();
        for (pos, opt) in math_buf.iter().enumerate() {
            if let Opt::Init(v) = opt {
                assert_eq!(buf_dst.data[pos], *v, "math/buf mismatch at dst wire position {pos}");
                live.push((pos, *v));
            }
        }
        // The dst's live cells (I=0 half) hold the source's I=1 slice: cells 2,3 (H=0), 6,7 (H=1), ...
        assert_eq!(live, vec![(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11)]);
    }

    /// `reduce` over a PADDED source mapping agrees between `MathStorage` and `BufStorage`. `BufStorage`
    /// reads the source by its `ReadPlan` base + live-residue deltas (the `# N` Top-pad residue positions
    /// dropped, never folded); this pins that the two backends stay equal when the source carries a `# N`
    /// Top pad (where a padded residue cell, if folded, would double-count its live sibling).
    #[test]
    fn reduce_padded_source_math_and_buf() {
        // m![Hp, Ep = 2 # 4] (Hp=3): a source padded on its inner axis (Ep live 2, wire extent 4).
        // reduce over the inner Ep, keeping m![Hp]: dst[h] = sum over the 2 live Ep cells of row h.
        let src = <m![Hp, Ep = 2 # 4]>::to_value();
        // Wire-order buffer (size = 3 * 4 = 12): each Hp row is [live0, live1, pad, pad]. `from_buf`
        // no-ops on the pad positions, so only the live cells (0,1 / 10,11 / 20,21) carry into reduce.
        let data: Vec<i32> = vec![0, 1, 0, 0, 10, 11, 0, 0, 20, 21, 0, 0];

        let math = MathStorage::<i32>::from_buf(&src, data.iter().copied());
        let math_out: MathStorage<i32> = math.reduce::<m![Hp, Ep = 2 # 4], m![Hp], _>(|a, b| a + b, 0, false);
        let math_res = math_out.into_buf(&<m![Hp]>::to_value());

        let buf = BufStorage::<i32>::from_buf(&src, data.iter().copied());
        let buf_out: BufStorage<i32> = buf.reduce::<m![Hp, Ep = 2 # 4], m![Hp], _>(|a, b| a + b, 0, false);
        let buf_res = buf_out.into_buf(&<m![Hp]>::to_value());

        assert_eq!(
            math_res, buf_res,
            "reduce over a padded source must agree across backends"
        );
        assert_eq!(math_res, vec![1, 21, 41]);
    }

    /// Size-shrinking `transmute` over a padded mapping, the MNIST `…tile(k).read().transmute()`
    /// shape: relabel a padded `Src` (`m![H, 1 # 2, E]`, the read output with the interleave slot as a
    /// `# 2` Top pad) down to the compact `Dst` (`m![H, E]`). `MathStorage` keeps its canonical
    /// (padding-free) buffer; `BufStorage` must COMPACT its physical buffer (drop the pad half) so both
    /// `into_buf` to the same compact values.
    #[test]
    fn transmute_compacts_padding_math_and_buf() {
        let src_map = <m![H, 1 # 2, E]>::to_value(); // size 12; live cells are the I=0 half
        let dst_map = <m![H, E]>::to_value(); // size 6, compact
        // Wire-order src buffer: each H row is [live0, live1, pad, pad]; `from_buf` no-ops the pads.
        let data: Vec<i32> = vec![0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0];

        // Simulation's transmute is a no-op (canonical storage already padding-free); read the
        // compact result straight out of the unchanged `MathStorage`.
        let math = MathStorage::<i32>::from_buf(&src_map, data.iter().copied());
        let math_res = math.into_buf(&dst_map);

        // Emulation's transmute compacts the physical buffer.
        let buf = BufStorage::<i32>::from_buf(&src_map, data.iter().copied());
        let buf_res = buf.transmute(&src_map, &dst_map).into_buf(&dst_map);

        assert_eq!(
            math_res, buf_res,
            "size-shrinking transmute must compact to the same values on both backends"
        );
        assert_eq!(math_res, vec![0, 1, 2, 3, 4, 5]);
    }

    /// Runs a non-zero-offset transpose on both `MathStorage` and `BufStorage` from the same parent
    /// buffer + offsets and asserts they write identical live cells (and that those live cells equal
    /// `expected_live`). `Src` is the source sub-view mapping, `Dst` the destination sub-view mapping;
    /// the destination parent buffer is sized to `Dst`.
    fn assert_offset_transpose_agrees<Parent: M, Src: M, Dst: M>(
        parent_data: &[i32],
        src_offset: &Index,
        dst_offset: &Index,
        expected_live: &[i32],
    ) {
        let parent_src = Parent::to_value();
        let dst_map = Dst::to_value();

        // Confirm the side under test really shifts the physical base off zero (a zero base would not
        // exercise the offset arithmetic at all).
        if *src_offset != Index::new() || *dst_offset != Index::new() {
            let wire_base = |mapping: &Mapping, offset: &Index| {
                let target = offset.clone().finalize();
                mapping
                    .indexes()
                    .into_iter()
                    .position(|i| i.finalize() == target)
                    .unwrap()
            };
            assert!(
                wire_base(&parent_src, src_offset) + wire_base(&dst_map, dst_offset) > 0,
                "offset case must shift the physical base off zero"
            );
        }

        let math = MathStorage::<i32>::from_buf(&parent_src, parent_data.iter().copied());
        let mut math_out = MathStorage::<i32>::uninit(&dst_map);
        math_out.transpose::<Src, Dst>(&math, src_offset, dst_offset, false);
        let math_buf = math_out.to_buf_opt(&dst_map);

        let buf = BufStorage::<i32>::from_buf(&parent_src, parent_data.iter().copied());
        let mut buf_out = buf_zeros(&dst_map);
        // Base (live-axis) maps: the src window resolves against the source parent, the dst window
        // against the dst sub-view (whose tiled axis stays a live resized `Symbol` here).
        buf_out.transpose::<Src, Dst>(&buf, src_offset, dst_offset, &parent_src, &dst_map, false);

        // Compare the live (non-padding) destination cells: `BufStorage` keeps raw `D` in padding
        // slots, `MathStorage` keeps `Opt::Uninit` there, so a live cell is where math is `Init`.
        let mut live = Vec::new();
        for (pos, opt) in math_buf.iter().enumerate() {
            if let Opt::Init(v) = opt {
                assert_eq!(buf_out.data[pos], *v, "math/buf mismatch at dst wire position {pos}");
                live.push(*v);
            }
        }
        assert_eq!(live, expected_live);
    }
}
