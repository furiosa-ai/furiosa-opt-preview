use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

use crate::backend::op_prep::{broadcast_axes, gather_params, scatter_params, transpose_broadcast};
use crate::cast::{Cast, ContractionCast};
use crate::scalar::*;
use crate::storage::par_iters::ParMappingExt;
use crate::storage::{DisjointScatter, PAR_MIN_JOB, min_cells_per_job, par_scatter};

/// `mapping`'s live offsets over `axes`, materialized (no shift). Only for the small per-cell index
/// sets (reduce's residue / fan, scatter/gather's key / residue); the large moves stay lazy.
fn live_offsets(mapping: &MappingValue, axes: &[AxisTerm]) -> Vec<usize> {
    mapping.iter(axes, &Index::new(), false).flatten().collect()
}

/// A `size`-long all-`Opt::Uninit` buffer, filled across the rayon pool. `Opt::Uninit` is not the
/// all-zeros bit pattern (`Opt` is a tagged enum), so a `vec![Uninit; size]` is a real per-element
/// store loop, not a zero-page alloc; on a large canvas that serial write is itself a bottleneck. A
/// rayon `collect` over the indexed `(0..size)` range pre-sizes the `Vec` (no reallocation) and writes
/// each slot on a worker; `with_min_len(PAR_MIN_JOB)` self-guards small sizes (a sub-threshold buffer
/// stays one effectively-serial job). The canvas for the disjoint-scatter relayouts, whose padding
/// cells are never written and so must start `Uninit`.
fn par_uninit_vec<D: Scalar>(size: usize) -> Vec<Opt<D>> {
    (0..size)
        .into_par_iter()
        .with_min_len(PAR_MIN_JOB)
        .map(|_| Opt::Uninit)
        .collect()
}

/// One read base per output cell of `out`, indexed by canonical output-buffer position. The base is
/// `out`'s offset into an operand buffer (`operand_axes`); the output coordinate, with the reduced /
/// contracted axes (absent from `out`) left at 0 for a per-cell delta to add.
///
/// `out.iter(operand_axes)` gives those offsets in the mapping's *structural* walk order, but the dense
/// output buffer is laid out in *canonical* (symbol-sorted, consolidated) order — the two differ once
/// axes are reordered or split. So reindex by `out.iter(out_axes)`, each cell's canonical position. A
/// padding output cell stays `None`. This is what lets the per-cell fold parallelize: each cell folds
/// independently into its own slot, no shared accumulator.
fn canonical_bases(
    out: &MappingValue,
    out_axes: &[AxisTerm],
    operand_axes: &[AxisTerm],
    out_size: usize,
) -> Vec<Option<usize>> {
    let mut bases = vec![None; out_size];
    // The two walks (`read` = operand offset, `pos` = canonical position) run in lockstep over the
    // same structural order. `pos` walks `out` over `out`'s OWN canonical axes into a buffer sized by
    // those axes, so each live cell visits its own canonical slot exactly once: `pos` is a bijection
    // over live cells (any fan lives in the *base value* -- many cells sharing one `read` -- never in
    // `pos`), hence the per-cell writes hit disjoint, in-range slots and split across the rayon pool.
    // This walk is O(out_size) -- for a broadcast reduce (out fanned across a broadcast axis) it is
    // `reduce`'s largest serial cost. `par_scatter`'s `with_min_len` keeps a small out serial.
    let reads = out.par_iter(operand_axes, &Index::new(), false);
    let positions = out.par_iter(out_axes, &Index::new(), false);
    // A live `pos` indexes `out`'s own canonical buffer, so it is in range by construction; feeding
    // the `(pos, read)` writes through `par_scatter` fails loud (its bound assert) in every build on a
    // planning bug, rather than dropping the cell. The distinct-index invariant is the `pos` bijection.
    let plan = (reads, positions)
        .into_par_iter()
        .map(|(read, pos)| pos.map(|pos| (pos, read)));
    par_scatter(&mut bases, plan);
    bases
}

/// Simulation tensor: canonical `ArrayD<Opt<D>>` host buffer with full ALU interpreter semantics.
///
/// `data` only stores the math (non-padding) values: its shape is the product of the `axes`' moduli.
/// `axes` is the mapping's canonical axis terms ([`Mapping::axes`]), symbol-sorted, sub-factors of
/// one symbol consolidated, so `data` is laid out in the natural canonical form, never wire order.
/// Every relayout (transpose / reduce / reshape / scatter / gather / the `into_buf` buffer seam) is a
/// move between canonical buffers driven by a lazy [`MappingExt::iter`] offset walk, no per-cell
/// coordinate decode.
#[derive(Clone, Debug)]
pub struct MathStorage<D: Scalar> {
    axes: Vec<AxisTerm>,
    data: ArrayD<Opt<D>>,
}

// Manual `PartialEq`/`Eq`: comparing only the stored axes + data (no mapping type to bound).
impl<D: Scalar> PartialEq for MathStorage<D> {
    fn eq(&self, other: &Self) -> bool {
        self.axes == other.axes && self.data == other.data
    }
}
impl<D: Scalar + Eq> Eq for MathStorage<D> {}

impl<D: Scalar> MathStorage<D> {
    fn new(axes: Vec<AxisTerm>, data: ArrayD<Opt<D>>) -> Self {
        Self { axes, data }
    }

    /// The dense canonical buffer as a flat slice (innermost axis contiguous). Infallible: every
    /// constructor builds `data` from a standard contiguous `ArrayD` (`from_shape_vec` / `from_elem` /
    /// `ArrayD::map`) and no method takes a non-contiguous view, so `as_slice` is always `Some`; the
    /// `expect` documents that invariant rather than guarding a reachable case.
    fn flat(&self) -> &[Opt<D>] {
        self.data.as_slice().expect("MathStorage buffer is contiguous")
    }

    fn flat_mut(&mut self) -> &mut [Opt<D>] {
        self.data.as_slice_mut().expect("MathStorage buffer is contiguous")
    }

    /// Element-wise map to a new scalar, `MathStorage`'s strategy: storage-native walk over the
    /// dense `ArrayD<Opt<D>>`, lifting the bare-scalar closure through `Opt` so an undefined cell
    /// stays undefined (`Uninit` propagates). The result keeps the same `axes`, so an
    /// elementwise op is mapping-free. Backs [`crate::backend::Backend::map`].
    pub(crate) fn map<D2: Scalar>(&self, f: impl Fn(D) -> D2 + Sync) -> MathStorage<D2> {
        // Cells are independent, so map them across the rayon pool over the dense canonical buffer.
        // The `collect` allocates the output uninitialized (no serial zero-init) and rayon's adaptive
        // split scales with the buffer. `Uninit` propagates per cell. Backs `contract_outer`'s widen
        // and the vector engine's unary ALU.
        let data: Vec<Opt<D2>> = self
            .flat()
            .par_iter()
            .map(|c| match c {
                Opt::Init(x) => Opt::Init(f(*x)),
                Opt::Uninit => Opt::Uninit,
            })
            .collect();
        self.with_flat(data)
    }

    /// Element-wise zip of two same-layout tensors by storage offset; the result is defined only
    /// where both inputs are (`Uninit` propagates). Backs [`crate::backend::Backend::zip_with`].
    pub(crate) fn zip_with<D2: Scalar, D3: Scalar>(
        &self,
        other: &MathStorage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> MathStorage<D3> {
        // Cells are independent; fold the pair across the rayon pool (see [`Self::map`]). Backs
        // `contract_outer`'s multiply and the vector engine's binary ALU.
        let data: Vec<Opt<D3>> = (self.flat(), other.flat())
            .into_par_iter()
            .map(|(x, y)| match (x, y) {
                (Opt::Init(x), Opt::Init(y)) => Opt::Init(f(*x, *y)),
                _ => Opt::Uninit,
            })
            .collect();
        self.with_flat(data)
    }

    /// Element-wise ternary zip; the result cell is defined only where all three inputs are.
    /// Ternary peer of [`Self::zip_with`].
    pub(crate) fn zip3_with<D2: Scalar, D3: Scalar, D4: Scalar>(
        &self,
        b: &MathStorage<D2>,
        c: &MathStorage<D3>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> MathStorage<D4> {
        let data: Vec<Opt<D4>> = (self.flat(), b.flat(), c.flat())
            .into_par_iter()
            .map(|(x, y, z)| match (x, y, z) {
                (Opt::Init(x), Opt::Init(y), Opt::Init(z)) => Opt::Init(f(*x, *y, *z)),
                _ => Opt::Uninit,
            })
            .collect();
        self.with_flat(data)
    }

    /// Rebuild a same-layout storage from a freshly mapped flat buffer (the elementwise ops keep
    /// `self`'s canonical `axes` and dense shape, only the scalar type changes).
    fn with_flat<D2: Scalar>(&self, data: Vec<Opt<D2>>) -> MathStorage<D2> {
        MathStorage::new(
            self.axes.clone(),
            ArrayD::from_shape_vec(self.data.raw_dim(), data).expect("elementwise op preserves the shape"),
        )
    }

    // Each relayout is one move: walk the read offsets (cells of `self`) and the write offsets (cells
    // of the dst) from `MappingExt::iter`, in lockstep, copying `src[read] -> dst[write]`. Nothing is
    // materialized. They differ only in which mapping/axes define the two offset streams and what the
    // inner step does, described on each op below.

    /// Writes a transposed/broadcast view of `src` into `self`. Backs
    /// [`crate::backend::Backend::transpose`]. Walk the dst cells; each reads the one src cell it maps
    /// from. Identical read/write plans collapse to a buffer copy.
    pub(crate) fn transpose<Src: M, Mapping: M>(
        &mut self,
        src: &MathStorage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        allow_broadcast: bool,
    ) {
        // Validate the relayout first (panics on an illegal broadcast / streaming-incompatible
        // decomposition). This must precede the fast path: `axes` resolves composites back to whole
        // symbols, so it cannot tell apart decompositions that share a wire order but tile incompatibly
        // (e.g. `A % 5` vs `A % 3`); only the full-mapping carve inside `transpose_broadcast` rejects those.
        transpose_broadcast::<Src, Mapping>(allow_broadcast);
        // Fast path. Same wire order, no shift: a plain copy (equal axes admit no broadcast). Chunk the
        // copy across the rayon pool -- a large DMA-passthrough dst (these dominate the call count) needs
        // more than one core to saturate memory. `par_chunks_mut(PAR_MIN_JOB)` self-guards small sizes
        // (a sub-threshold copy is a single chunk = one plain `memcpy`, no meaningful rayon overhead).
        if self.axes == src.axes && *src_offset == Index::new() && *dst_offset == Index::new() {
            let src_flat = src.flat();
            self.flat_mut()
                .par_chunks_mut(PAR_MIN_JOB)
                .zip(src_flat.par_chunks(PAR_MIN_JOB))
                .for_each(|(dst_chunk, src_chunk)| dst_chunk.copy_from_slice(src_chunk));
            return;
        }
        let dst_mapping = Mapping::to_value();
        let reads = dst_mapping.par_iter(&src.axes, src_offset, false);
        let writes = dst_mapping.par_iter(&self.axes, dst_offset, false);
        let src_flat = src.flat();
        let dst_len = self.flat().len();
        // The plan is a bijection: each live dst cell is the `write` of exactly one step (padding /
        // out-of-range steps write nothing), so the per-step scatters target disjoint slots and run in
        // parallel. `par_iter` splits each mapping walk by seeking (`ParMappingProducer`); the tuple
        // multizip pairs the two in lockstep. A padding step (`None`) or a split-then-padded slot past
        // the dense buffer (`write >= dst_len`) skips (yields `None`); the rest copy `src[read] -> dst[write]`.
        let plan = (reads, writes).into_par_iter().map(|(read, write)| {
            let (Some(read), Some(write)) = (read, write) else {
                return None;
            };
            (write < dst_len)
                .then(|| src_flat.get(read).map(|&cell| (write, cell)))
                .flatten()
        });
        par_scatter(self.flat_mut(), plan);
    }

    /// Reduces the factors of `self`'s mapping that are absent in `Dst`, returning the result. Each
    /// output cell independently folds its block of the reduced (residue) axes, so the per-cell
    /// reduction parallelizes with no contention. Backs [`crate::backend::Backend::reduce`].
    ///
    /// Each output cell's source read base is the cell's coordinate projected onto the source (an axis
    /// `Dst` adds beyond the reduced source is a broadcast, absent from the source ⇒ many output cells
    /// share a base); the residue deltas are the reduced axes, shared across all cells. So a read is the
    /// additive `base + residue_delta` over the canonical buffer. `Opt::Uninit` propagates through the
    /// fold (`zip_map`).
    pub(crate) fn reduce<Src: M, Dst: M, R: Fn(D, D) -> D + Sync>(
        &self,
        reduce_fn: R,
        identity: D,
        allow_broadcast: bool,
    ) -> Self {
        let src = Src::to_value();
        let dst = Dst::to_value();
        // Axes `Dst` adds beyond the reduced source are broadcast; `!allow_broadcast` rejects them.
        let broadcast = broadcast_axes(&src, &dst);
        assert!(
            allow_broadcast || broadcast.axes().is_empty(),
            "reduce: Dst adds axes absent from the source; pass allow_broadcast=true for reduce-then-broadcast"
        );
        // The reduced axes are the source's beyond the kept output (`inter`); their per-coordinate
        // offsets are the residue, shared across output cells. A broadcast axis (in `dst`, not the
        // source) projects to 0, so its fan copies share one read base and each recomputes the fold.
        let inter = dst.carve(&broadcast);
        let reduce_residue = src.carve(&inter);
        let out_axes = dst.axes();
        let shape: Vec<usize> = out_axes.iter().map(|term| term.modulo).collect();
        let out_size = shape.iter().product();
        let base = canonical_bases(&dst, &out_axes, &self.axes, out_size);
        let residue = live_offsets(&reduce_residue, &self.axes);
        let src_flat = self.flat();
        // One output cell folds its residue block. Output cells are independent, so fold each across the
        // rayon pool and `collect` directly -- no serial `vec![Uninit; out_size]` pre-fill (each cell is
        // written once; a dead/padding cell returns `Opt::Uninit`). A split-then-padded reduced axis can
        // land a residue read past the dense buffer, so `get` skips an out-of-range one (not a real
        // cell). `with_min_len` keeps each job at >= `PAR_MIN_JOB` work (cells x residue) so small
        // reduces stay one job.
        let min_cells = min_cells_per_job(residue.len());
        let out_data: Vec<Opt<D>> = (0..out_size)
            .into_par_iter()
            .with_min_len(min_cells)
            .map(|i| {
                let Some(read_base) = base[i] else { return Opt::Uninit };
                let mut acc = Opt::Init(identity);
                for &delta in &residue {
                    if let Some(&v) = src_flat.get(read_base + delta) {
                        acc = acc.zip_map(v, &reduce_fn);
                    }
                }
                acc
            })
            .collect();
        Self::new(
            out_axes,
            ArrayD::from_shape_vec(IxDyn(&shape), out_data).expect("plan yields the canonical shape"),
        )
    }

    /// Fused contraction (generalized matmul): each output cell independently sums `lhs * rhs` over
    /// the contracted axes (those in `Union` but absent from `Out`). O(`Out`) memory, no `Union` outer
    /// product materialized. Backs [`crate::backend::Backend::contraction`] for Simulation.
    ///
    /// Output cells are mutually independent, so the per-cell reduction parallelizes with no
    /// contention (above a work threshold). Equivalent to
    /// `lhs.transpose::<Union>(true) * rhs.transpose::<Union>(true)` reduced onto `Out`: each operand
    /// offset splits into the output cell's base plus a contracted-coordinate delta (the additive
    /// canonical-buffer model `reduce` uses). `Opt::Uninit` propagates through both the product and the
    /// running sum (each a `zip_map` short-circuiting to `Uninit`), matching that composition.
    ///
    /// Needs no `Lhs` / `Rhs` type params (unlike `BufStorage`): each operand storage already carries
    /// its canonical `axes`, which the output/contracted mappings project onto directly.
    pub(crate) fn contraction<Union: M, Out: M>(lhs: &MathStorage<D>, rhs: &MathStorage<D>) -> MathStorage<D>
    where
        D: ContractionCast,
    {
        let union = Union::to_value();
        let out = Out::to_value();
        let out_axes = out.axes();
        let shape: Vec<usize> = out_axes.iter().map(|term| term.modulo).collect();
        let out_size = shape.iter().product();
        // The contracted axes are the source's beyond the kept output, exactly `reduce`'s residue.
        let contracted = union.carve(&out);
        // One base offset per output cell (the output coordinate, contracted at 0), plus per contracted
        // coordinate the operand offset deltas shared across all cells. Bases are indexed by canonical
        // output position; deltas just walk the contracted axes.
        let lhs_base = canonical_bases(&out, &out_axes, &lhs.axes, out_size);
        let rhs_base = canonical_bases(&out, &out_axes, &rhs.axes, out_size);
        let lhs_delta = live_offsets(&contracted, &lhs.axes);
        let rhs_delta = live_offsets(&contracted, &rhs.axes);
        let lhs_flat = lhs.flat();
        let rhs_flat = rhs.flat();
        // One output cell: sum the products over the contracted block in `<D as ContractionCast>::Output`.
        // Each operand widens on load via `Cast` (`D -> Output`); the sum narrows via `Cast` (`Output -> D`).
        // A split-then-padded contracted coordinate can land past a dense buffer, so skip an out-of-range
        // projection (as the relayouts do); valid contractions never produce one.
        //
        // The `Opt` tag is hoisted out of the per-element fold: a raw `Output` accumulator runs the MACs
        // unwrapped, and `uninit` records whether any in-range operand cell was `Opt::Uninit` (which the
        // wrapped fold propagated to the whole cell via `zip_map`'s short-circuit). This is bit-identical
        // to the wrapped fold -- same widen, same fold order over the `Init` prefix, same `+`, same final
        // narrow -- with the per-MAC tag check lifted to a single branch instead of two `zip_map`s an
        // element. Once `uninit` is set the arithmetic stops (matching the wrapped fold, whose `zip_map`
        // short-circuited every later `+`), so no spurious overflow past the first uninitialized operand.
        let cell = |i: usize| {
            let (Some(lhs_base), Some(rhs_base)) = (lhs_base[i], rhs_base[i]) else {
                return Opt::Uninit;
            };
            let mut acc = <<D as ContractionCast>::Output as num_traits::Zero>::zero();
            for (&lhs_delta, &rhs_delta) in lhs_delta.iter().zip(&rhs_delta) {
                if let (Some(&l), Some(&r)) = (lhs_flat.get(lhs_base + lhs_delta), rhs_flat.get(rhs_base + rhs_delta)) {
                    // An out-of-range MAC is skipped (as the wrapped fold's `if let` did); an in-range MAC
                    // with an `Uninit` operand makes the whole cell `Uninit` -- return early, matching the
                    // wrapped fold that short-circuited every later step once `acc` went `Uninit`.
                    let (Opt::Init(l), Opt::Init(r)) = (l, r) else {
                        return Opt::Uninit;
                    };
                    let l: <D as ContractionCast>::Output = Cast::cast(l);
                    let r: <D as ContractionCast>::Output = Cast::cast(r);
                    acc = acc + l * r;
                }
            }
            Opt::Init(Cast::cast(acc))
        };
        // Output cells are independent, so map the per-cell reduction across the rayon pool and
        // `collect` directly into the output buffer -- no serial `vec![Uninit; out_size]` pre-fill
        // (every cell is written exactly once, so the pre-fill was pure wasted serial memset).
        // `with_min_len` holds each job at >= `PAR_MIN_JOB` work (cells x contracted): a small
        // contraction stays one (effectively serial) job rather than fragmenting, while rayon splits
        // larger ones across cores.
        let min_cells = min_cells_per_job(lhs_delta.len());
        let out_data: Vec<Opt<D>> = (0..out_size)
            .into_par_iter()
            .with_min_len(min_cells)
            .map(cell)
            .collect();
        Self::new(
            out_axes,
            ArrayD::from_shape_vec(IxDyn(&shape), out_data).expect("plan yields the canonical shape"),
        )
    }

    /// Scatters `self` into `dst` at positions read from the `i32` index tensor. Backs
    /// [`crate::backend::Backend::scatter`]. Mirror of [`Self::gather`]: the runtime index drives the
    /// dst (write) side, the key cells are the static src (read) residue.
    pub(crate) fn scatter<Src: M, Key: M, Dst: M, Idx: M>(
        &self,
        dst: &mut MathStorage<D>,
        index: &MathStorage<i32>,
        scaled: bool,
    ) {
        let key = Key::to_value();
        let (payload, dst_term) = scatter_params(&Src::to_value(), &Dst::to_value(), &key);
        let indices = decode_coord_indices::<Idx>(index, decode_stride::<D>(&payload, scaled));
        let read_inner = live_offsets(&key, &self.axes);
        let write_inner = index_offsets(&MappingValue::from_terms(vec![dst_term.to_term()]), &dst.axes, &indices);
        self.indexed_move(dst, &payload, &read_inner, &write_inner);
    }

    /// Gathers from `self` (table) into `dst` at positions read from the index tensor. Backs
    /// [`crate::backend::Backend::gather`]. Mirror of [`Self::scatter`]: the runtime index drives the
    /// src (read) side, the dst-residue cells are the static dst (write) residue.
    pub(crate) fn gather<Src: M, Dst: M, Idx: M>(
        &self,
        dst: &mut MathStorage<D>,
        index: &MathStorage<i32>,
        scaled: bool,
    ) {
        let params = gather_params(&Src::to_value(), &Dst::to_value(), &Idx::to_value());
        let indices = decode_coord_indices::<Idx>(index, decode_stride::<D>(&params.payload, scaled));
        let read_inner = index_offsets(
            &MappingValue::from_terms(vec![params.src_term.to_term()]),
            &self.axes,
            &indices,
        );
        let write_inner = live_offsets(&params.idx_residue, &dst.axes);
        // Gather's write side is duplicate-free (the payload walk is a bijection over `dst`, the
        // `write_inner` residue is distinct), so the per-payload-cell scatters target disjoint slots
        // and parallelize; the index only duplicates *reads*, which is harmless.
        self.indexed_gather(dst, &params.payload, &read_inner, &write_inner);
    }

    /// Moves cells from `self` to `dst` over the shared `payload`: walk the payload once (its src-read
    /// and dst-write offsets paired), and for each payload cell move every free cell, adding the
    /// per-free-cell `read_inner` / `write_inner` offsets. The indexed side's offsets come from the
    /// runtime index (see [`index_offsets`]); the other side's are the static residue. Both inner
    /// vectors run over the same free dimension.
    ///
    /// The serial form, used by `scatter`: its `write_inner` is index-driven, so two source cells can
    /// target the SAME dst slot (a duplicate scatter index), and HW's last-writer-wins order would not
    /// survive a parallel run. `gather` (disjoint writes) uses [`Self::indexed_gather`] instead.
    fn indexed_move(&self, dst: &mut Self, payload: &MappingValue, read_inner: &[usize], write_inner: &[usize]) {
        let payload_reads = payload.iter(&self.axes, &Index::new(), false);
        let payload_writes = payload.iter(&dst.axes, &Index::new(), false);
        let src_flat = self.flat();
        let dst_flat = dst.flat_mut();
        for (read_base, write_base) in payload_reads.zip(payload_writes) {
            let (Some(read_base), Some(write_base)) = (read_base, write_base) else {
                continue;
            };
            for (&read_off, &write_off) in read_inner.iter().zip(write_inner) {
                dst_flat[write_base + write_off] = src_flat[read_base + read_off];
            }
        }
    }

    /// Parallel peer of [`Self::indexed_move`] for `gather`: the dst-write side is duplicate-free, so
    /// the payload-cell steps scatter into disjoint slots and run across the rayon pool (the same
    /// raw-pointer disjoint scatter [`Self::transpose`] uses; safe Rust cannot express "disjoint
    /// indices of one `&mut [_]`"). A duplicate runtime index only repeats a *read*, never a write.
    fn indexed_gather(&self, dst: &mut Self, payload: &MappingValue, read_inner: &[usize], write_inner: &[usize]) {
        let payload_reads = payload.par_iter(&self.axes, &Index::new(), false);
        let payload_writes = payload.par_iter(&dst.axes, &Index::new(), false);
        let src_flat = self.flat();
        let dst_len = dst.flat().len();
        // Each payload step scatters an inner run of cells (not one), so this keeps the explicit rayon
        // loop rather than routing through `par_scatter` (a single write per step); the scatter itself
        // still goes through the shared bounds-checked `DisjointScatter`. `with_min_len` keeps a small
        // gather one job. A split-then-padded slot past the dense buffer (`write >= dst_len`) skips.
        let scatter = DisjointScatter::new(dst.flat_mut());
        let scatter = &scatter; // capture the `Sync` wrapper by reference, not the bare `*mut`
        (payload_reads, payload_writes)
            .into_par_iter()
            .with_min_len(PAR_MIN_JOB)
            .for_each(|(read_base, write_base)| {
                let (Some(read_base), Some(write_base)) = (read_base, write_base) else {
                    return;
                };
                for (&read_off, &write_off) in read_inner.iter().zip(write_inner) {
                    let (read, write) = (read_base + read_off, write_base + write_off);
                    if let (Some(&cell), true) = (src_flat.get(read), write < dst_len) {
                        // SAFETY: gather's write plan is a bijection (disjoint payload-walk base +
                        // distinct residue), so every live `write` is distinct across steps -- no two
                        // parallel jobs touch one slot. `DisjointScatter::write` bounds `write`.
                        unsafe { scatter.write(write, cell) };
                    }
                }
            });
    }

    /// Reinterprets the physical buffer from `Mapping`-shaped to `Mapping2`-shaped, returning the
    /// result. Backs [`crate::backend::Backend::reshape`]. The wire buffer is shared between the two
    /// layouts, so walk each side's wire plan and copy src canonical cell to dst canonical cell.
    pub(crate) fn reshape<Mapping: M, Mapping2: M>(&self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        let dst_axes = Mapping2::to_value().axes();
        let reads = Mapping::to_value().par_iter(&self.axes, &Index::new(), true);
        let writes = Mapping2::to_value().par_iter(&dst_axes, &Index::new(), true);
        let shape: Vec<usize> = dst_axes.iter().map(|term| term.modulo).collect();
        let src_flat = self.flat();
        let mut data = par_uninit_vec::<D>(shape.iter().product());
        let dst_len = data.len();
        // The two wire plans have the same length (`Mapping::SIZE == Mapping2::SIZE`) and share the
        // physical position order; each live position pairs one src read with one dst write. The dst
        // write plan is a bijection over live cells (distinct wire positions -> distinct canonical
        // slots), so the per-position writes target disjoint slots and run across the rayon pool. A
        // split-then-padded axis can land a read/write past the dense buffer (the same out-of-range case
        // `transpose` guards); a `None` position or `read`/`write` out of range skips (yields `None`),
        // as the serial predecessor's `data[write] = src_flat[read]` did only for in-range cells.
        let plan = (reads, writes).into_par_iter().map(|(read, write)| {
            let (Some(read), Some(write)) = (read, write) else {
                return None;
            };
            (write < dst_len)
                .then(|| src_flat.get(read).map(|&cell| (write, cell)))
                .flatten()
        });
        par_scatter(&mut data, plan);
        Self::new(
            dst_axes,
            ArrayD::from_shape_vec(IxDyn(&shape), data).expect("plan yields the canonical shape"),
        )
    }

    /// Construct from a logical `Opt<D>` buffer in physical layout. Simulation-only Opt API.
    pub(crate) fn from_opt_buf(mapping: &MappingValue, data: impl IntoIterator<Item = Opt<D>>) -> Self {
        // A padding position has no dst slot, so its value (typically `Opt::Uninit`) is dropped,
        // matching the padding-stripped canonical `data`.
        Self::scatter_buf(mapping, data.into_iter().collect())
    }

    /// Scatter a physical-layout buffer (`mapping`-order, padding included) into a fresh canonical
    /// `data` by walking the wire offset plan lazily, each physical position's canonical dst slot,
    /// no per-position `finalize` and no offset array. Shared by `from_buf` / `from_opt_buf`.
    fn scatter_buf(mapping: &MappingValue, data: Vec<Opt<D>>) -> Self {
        let axes = mapping.axes();
        let shape: Vec<usize> = axes.iter().map(|term| term.modulo).collect();
        // Each physical value lands at its wire offset; padding cells (`None`) have no canonical slot.
        let mut scattered = par_uninit_vec::<D>(shape.iter().product());
        let dst_len = scattered.len();
        // The wire plan maps each *live* physical position to a distinct canonical slot (padding
        // positions map to `None`), so the per-position writes target disjoint slots and run across
        // the rayon pool. `par_iter_cells` yields one offset per physical position in position order,
        // so its enumerated index selects the matching `data[position]`. `position < data.len()`
        // mirrors the serial `.zip(&data)` truncation; `slot < dst_len` guards a split-then-padded slot
        // past the dense buffer (as `transpose` does); either miss skips (yields `None`).
        let data = &data;
        let plan = mapping.par_iter_cells().enumerate().map(|(position, offset)| {
            let (Some(slot), Some(&value)) = (offset, data.get(position)) else {
                return None;
            };
            (slot < dst_len).then_some((slot, value))
        });
        par_scatter(&mut scattered, plan);
        let data = ArrayD::from_shape_vec(IxDyn(&shape), scattered).expect("plan yields the canonical shape");
        Self::new(axes, data)
    }

    /// Validates the input length matches `mapping.size()` then constructs via
    /// [`Self::from_opt_buf`]. Simulation-only Opt API.
    pub(crate) fn try_from_opt_buf(
        mapping: &MappingValue,
        data: impl IntoIterator<Item = Opt<D>>,
    ) -> Result<Self, crate::tensor::BufferConvertError> {
        let data: Vec<Opt<D>> = data.into_iter().collect();
        if data.len() != mapping.size() {
            return Err(crate::tensor::BufferConvertError::length_mismatch(
                mapping.size(),
                data.len(),
            ));
        }
        Ok(Self::from_opt_buf(mapping, data))
    }

    /// Serialize to a logical `Opt<D>` buffer in physical layout. Simulation-only Opt API. The
    /// canonical `data` is gathered into wire order through the projection, one offset per physical
    /// position, no per-position `finalize`.
    pub(crate) fn to_buf_opt(&self, mapping: &MappingValue) -> Vec<Opt<D>> {
        // Gather each physical position's value through the wire offsets; a padding cell reads `Uninit`.
        // Output is indexed by physical position (one cell per wire offset), so the gather is an
        // embarrassingly-parallel indexed map across the rayon pool (`par_iter_cells` splits the walk
        // by seeking). `with_min_len` keeps a small buffer a single serial job. The `flat[slot]` index
        // is unchecked-by-design: `slot` is a live offset the mapping projects onto `self.axes` (this
        // storage's own canonical layout), so it is in range of `flat` by construction.
        let flat = self.flat();
        mapping
            .par_iter(&self.axes, &Index::new(), true)
            .with_min_len(PAR_MIN_JOB)
            .map(|offset| offset.map_or(Opt::Uninit, |slot| flat[slot]))
            .collect()
    }

    /// Build from a flat `D` buffer in `mapping`-order.
    pub(crate) fn from_buf(mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self {
        Self::scatter_buf(mapping, data.into_iter().map(Opt::Init).collect())
    }

    /// Serialize to a flat `D` buffer in physical (wire) order. The dense `data` is canonical
    /// (`axes()`-ordered), so the physical buffer is gathered through the projection
    /// (physical-position → canonical-offset map).
    pub(crate) fn into_buf(self, mapping: &MappingValue) -> Vec<D> {
        self.to_buf_opt(mapping)
            .into_iter()
            .map(|value| match value {
                Opt::Init(value) => value,
                Opt::Uninit => panic!(
                    "MathStorage::into_buf called on a tensor containing Opt::Uninit slots; \
                     use the logical Opt-buffer view instead."
                ),
            })
            .collect()
    }

    /// An uninitialized tensor of this layout.
    pub(crate) fn uninit(mapping: &MappingValue) -> Self {
        // Store the canonical axis terms (`Mapping::axes()`): the dense `data` is the natural
        // canonical form, and the buffer seam (`into_buf`) re-permutes to wire order via its plan.
        let axes = mapping.axes();
        // Dense canonical shape: each axis's modulus (padding-free, padding lives only in the wire
        // buffer materialized at `into_buf`).
        let shape: Vec<usize> = axes.iter().map(|term| term.modulo).collect();
        // A serial `ArrayD::from_elem` memset of the whole dense buffer dominates the per-op cost
        // (every transpose / reduce / contraction allocates a fresh `uninit` dst first), so fill the
        // canvas across the rayon pool via the shared [`par_uninit_vec`].
        let buf = par_uninit_vec::<D>(shape.iter().product());
        let data = ArrayD::from_shape_vec(IxDyn(&shape), buf).expect("uninit shape matches its element count");
        Self::new(axes, data)
    }
}

/// The divisor [`decode_coord_indices`] applies: a scaled index tensor holds byte offsets into
/// payload-sized blocks (divide by the block's byte size, padding excluded); an unscaled one holds
/// element indices.
fn decode_stride<D>(payload: &MappingValue, scaled: bool) -> usize {
    if scaled {
        payload.clone().remove_padding().size() * std::mem::size_of::<D>()
    } else {
        1
    }
}

/// Decodes a coordinate-addressed (`MathStorage`-backed) `i32` index tensor's `Idx::SIZE` positions
/// to element positions (each value divided by `index_stride`). Reads by logical coordinate so it
/// stays view-correct. Shared by scatter and gather.
fn decode_coord_indices<Idx: M>(index: &MathStorage<i32>, index_stride: usize) -> Vec<usize> {
    // Gather the index tensor into `Idx` wire order through its offset plan (no per-position
    // `finalize`), then scale each value to an element position.
    index
        .to_buf_opt(&Idx::to_value())
        .into_iter()
        .map(|value| {
            let Opt::Init(v) = value else {
                panic!("scatter/gather index must be initialized")
            };
            usize::try_from(v).expect("scatter/gather index must be non-negative") / index_stride
        })
        .collect()
}

/// The per-free-cell offsets of the indexed `axis` (within `axes`) for a runtime index vector: the
/// position → offset table ([`live_offsets`]) looked up at each index value. The indexed
/// side of `scatter` / `gather`.
fn index_offsets(axis: &MappingValue, axes: &[AxisTerm], indices: &[usize]) -> Vec<usize> {
    let positions = live_offsets(axis, axes);
    indices.iter().map(|&i| positions[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::MathStorage;
    use crate::scalar::{Opt, bf16};
    use furiosa_mapping::*;

    /// `MathStorage::contraction` reproduces a plain matmul. `[M,K] · [K,N] -> [M,N]` with no padding;
    /// `into_buf` returns the result in row-major `[M,N]` wire order, hand-checkable.
    #[test]
    fn contraction_matches_matmul() {
        axes![M = 2, K = 3, N = 2];
        // lhs row-major [[1,2,3],[4,5,6]]; rhs row-major [[1,2],[3,4],[5,6]].
        let lhs = MathStorage::<f32>::from_buf(&<m![M, K]>::to_value(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = MathStorage::<f32>::from_buf(&<m![K, N]>::to_value(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = MathStorage::contraction::<m![M, K, N], m![M, N]>(&lhs, &rhs);
        // [[22,28],[49,64]].
        assert_eq!(out.into_buf(&<m![M, N]>::to_value()), vec![22.0f32, 28.0, 49.0, 64.0]);
    }

    /// Same product, but `Out` is written `[N, M]` so the mapping's structural walk order differs from
    /// the canonical (symbol-sorted `[M, N]`) buffer order. Guards `canonical_bases`: indexing the per
    /// cell bases by structural position instead would scramble the result.
    #[test]
    fn contraction_reordered_out_axes() {
        axes![M = 2, K = 3, N = 2];
        let lhs = MathStorage::<f32>::from_buf(&<m![M, K]>::to_value(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = MathStorage::<f32>::from_buf(&<m![K, N]>::to_value(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = MathStorage::contraction::<m![M, K, N], m![N, M]>(&lhs, &rhs);
        // The (M,N) = [[22,28],[49,64]] product read back N-major.
        assert_eq!(out.into_buf(&<m![N, M]>::to_value()), vec![22.0f32, 49.0, 28.0, 64.0]);
    }

    /// A `bf16` `K = 256` contraction accumulates in `f32`, recovering small addends a narrow per-step
    /// accumulator drops: `lhs = [1.0, 1/128, .., 1/128]` against `rhs = ones` (both exact in bf16, so the
    /// only error is accumulation rounding) sums to `1.0 + 255*(1/128)`. A bf16 running sum in serial fold
    /// order (`1.0` first) saturates near `2.0` and swallows every later `1/128`; the `f32` accumulator
    /// keeps them. (A reassociated/tree bf16 sum could reach `~3.0` and slip the band, so this
    /// discriminates only because the fold here is serial within the cell.)
    #[test]
    fn contraction_bf16_recovers_small_addends_via_wide_accumulator() {
        axes![M = 1, K = 256, N = 1];
        let lhs: Vec<bf16> = std::iter::once(1.0)
            .chain(std::iter::repeat_n(1.0 / 128.0, 255))
            .map(bf16::from_f32)
            .collect();
        let rhs = vec![bf16::from_f32(1.0); 256];
        let lhs = MathStorage::<bf16>::from_buf(&<m![M, K]>::to_value(), lhs);
        let rhs = MathStorage::<bf16>::from_buf(&<m![K, N]>::to_value(), rhs);
        let out = MathStorage::contraction::<m![M, K, N], m![M, N]>(&lhs, &rhs);
        let got = out.into_buf(&<m![M, N]>::to_value())[0].to_f32();
        // Exact f32 reference 1.0 + 255*(1/128); bf16 has 7 mantissa bits so 1 ulp = reference * 2^-7.
        let reference = 2.9921875_f32;
        assert!(
            (got - reference).abs() <= reference * f32::exp2(-7.0),
            "f32 accumulator should recover small addends bf16 drops: got={got}"
        );
    }

    /// The integer counterpart, end-to-end through the storage fold: an `i16` `K = 256` dot of
    /// `1000 * 1000` sums to `256_000_000`, far past `i16`'s range. Accumulating in the narrow `i16` would
    /// wrap at every step; accumulating in the `i32` `ContractionCast::Output` holds the true sum and
    /// wraps `as i16` once at the narrow, so the output equals `256_000_000 as i16` (the value
    /// `cast::tests::narrow_integer_accumulates_in_i32` pins, here through `MathStorage::contraction`).
    #[test]
    fn contraction_i16_accumulates_wide_then_wraps_once() {
        axes![M = 1, K = 256, N = 1];
        let lhs = MathStorage::<i16>::from_buf(&<m![M, K]>::to_value(), vec![1000i16; 256]);
        let rhs = MathStorage::<i16>::from_buf(&<m![K, N]>::to_value(), vec![1000i16; 256]);
        let out = MathStorage::contraction::<m![M, K, N], m![M, N]>(&lhs, &rhs);
        assert_eq!(out.into_buf(&<m![M, N]>::to_value())[0], 256_000_000i32 as i16);
    }

    /// Pins that the rayon-parallel `f32` Simulation contraction equals a SERIAL per-cell fold in the
    /// same contracted order, bit-for-bit. `f32` add is non-associative, so this is the real witness
    /// that each output cell's sum is grouped independently of the worker split (a reassociating
    /// implementation would diverge from the serial reference here); a mere run-to-run rerun would not,
    /// since the split is data-independent. The serial reference walks the canonical `[M,K]`/`[K,N]`
    /// row-major buffers directly. (Emulation's `BufStorage` reassociates and is deliberately NOT
    /// pinned this way; only Simulation promises serial-equivalent bit-exactness.)
    #[test]
    fn contraction_f32_matches_serial_fold_bit_for_bit() {
        // Dims: M*N = 4096 output cells (> PAR_MIN_JOB), K = 257 makes each cell a real fold.
        let (rows, contracted, cols) = (64usize, 257usize, 64usize);
        axes![M = 64, K = 257, N = 64];
        let lhs_f: Vec<f32> = (0..rows * contracted).map(|i| (i % 17) as f32 * 0.3 - 2.0).collect();
        let rhs_f: Vec<f32> = (0..contracted * cols).map(|i| (i % 11) as f32 * 0.7 - 1.0).collect();
        let lhs = MathStorage::<f32>::from_buf(&<m![M, K]>::to_value(), lhs_f.clone());
        let rhs = MathStorage::<f32>::from_buf(&<m![K, N]>::to_value(), rhs_f.clone());
        let got = MathStorage::contraction::<m![M, K, N], m![M, N]>(&lhs, &rhs).into_buf(&<m![M, N]>::to_value());
        // Serial reference: each output cell folded left-to-right over K, the same order the per-cell
        // parallel fold uses; bit-exact only if the parallel path does not reassociate across workers.
        let serial: Vec<f32> = (0..rows * cols)
            .map(|cell| {
                let (m, n) = (cell / cols, cell % cols);
                (0..contracted).fold(0.0f32, |acc, k| acc + lhs_f[m * contracted + k] * rhs_f[k * cols + n])
            })
            .collect();
        assert_eq!(
            got, serial,
            "parallel f32 contraction must match the serial per-cell fold bit-for-bit"
        );
    }

    /// The fold seed and `Uninit` propagation. The contracted residue is never empty (the no-axes
    /// case is one zero-offset term, a single MAC), so a contracted axis of extent 1 already exercises
    /// the seed path: `acc` starts `Init(zero())`, takes one product, and narrows once. A genuinely
    /// uninitialized operand cell makes its product, and thus the whole output cell, `Uninit` -- it
    /// short-circuits the per-MAC `zip_map` rather than narrowing the seed to `0`.
    #[test]
    fn contraction_seed_and_uninit_propagation() {
        // K = 1: one MAC over the seed. Output is the plain product, never the seed `0`.
        axes![A = 1, K1 = 1, B = 1];
        let lhs = MathStorage::<f32>::from_buf(&<m![A, K1]>::to_value(), [3.0]);
        let rhs = MathStorage::<f32>::from_buf(&<m![K1, B]>::to_value(), [4.0]);
        let out = MathStorage::contraction::<m![A, K1, B], m![A, B]>(&lhs, &rhs);
        assert_eq!(out.into_buf(&<m![A, B]>::to_value()), vec![12.0f32]);

        // A single uninitialized operand cell makes its product (and thus its output cell) `Uninit`,
        // not the narrowed seed `0`.
        axes![K = 2];
        let lhs = MathStorage::<f32>::from_opt_buf(&<m![A, K]>::to_value(), [Opt::Init(1.0), Opt::Uninit]);
        let rhs = MathStorage::<f32>::from_buf(&<m![K, B]>::to_value(), [1.0, 1.0]);
        let out = MathStorage::contraction::<m![A, K, B], m![A, B]>(&lhs, &rhs);
        assert_eq!(out.to_buf_opt(&<m![A, B]>::to_value()), vec![Opt::Uninit]);
    }

    use super::par_uninit_vec;

    /// `reduce` with `allow_broadcast=true` fans the reduced result across an added broadcast axis --
    /// the case `canonical_bases`'s parallel walk targets (many output cells sharing one read base, the
    /// `pos` bijection over live cells still holding). Every broadcast copy must equal the reduced value.
    /// Sized so the output (`A * B * C`) exceeds `PAR_MIN_JOB`, forcing `canonical_bases`'s disjoint
    /// scatter to actually split across workers rather than run as one serial job.
    #[test]
    fn reduce_then_broadcast_fans_out() {
        // A * B * C = 3 * 4 * 6000 = 72000 > PAR_MIN_JOB (65536): the broadcast fan drives out_size past
        // the split threshold, so the parallel `canonical_bases` write plan runs multi-job here.
        axes![A = 3, B = 4, C = 6000];
        let (a_ext, b_ext, c_ext) = (3usize, 4, 6000);
        // src[a,b] = a*4+b; reduce nothing structural but broadcast-add axis C: out[a,b,c] = src[a,b].
        let src = MathStorage::<i32>::from_buf(&<m![A, B]>::to_value(), (0..12).collect::<Vec<_>>());
        let out = src.reduce::<m![A, B], m![A, B, C], _>(|a, b| a + b, 0, true);
        let got = out.into_buf(&<m![A, B, C]>::to_value());
        // Oracle: independent nested loop, out[a,b,c] = src[a,b] for every c.
        let mut expected = Vec::with_capacity(a_ext * b_ext * c_ext);
        for a in 0..a_ext as i32 {
            for b in 0..b_ext as i32 {
                for _c in 0..c_ext {
                    expected.push(a * 4 + b);
                }
            }
        }
        assert_eq!(got, expected);
    }

    /// `reshape` relabels the wire buffer to a differently-factored mapping of the same SIZE. Round-trips
    /// a composite through a transposed composite and back; the disjoint-scatter write plan must land
    /// every live cell. Hand-checked oracle (identity reshape is a no-op relabel here).
    #[test]
    fn reshape_relabels_wire_buffer() {
        axes![A = 3, B = 4];
        let src = MathStorage::<i32>::from_buf(&<m![A, B]>::to_value(), (0..12).collect::<Vec<_>>());
        // Reshape [A,B] -> [A,B] is an identity relabel (same wire order, same axes): the buffer is
        // unchanged, so into_buf reproduces the input.
        let out = src.reshape::<m![A, B], m![A, B]>();
        assert_eq!(out.into_buf(&<m![A, B]>::to_value()), (0..12).collect::<Vec<_>>());
    }

    /// A split-sized (> PAR_MIN_JOB) reshape actually exercises the rayon multi-job split (every smaller
    /// test runs the parallel path as a single serial job). Identity relabel, so the oracle is the input.
    #[test]
    fn reshape_large_split_matches_input() {
        axes![Big = 200000]; // > PAR_MIN_JOB (65536) forces the disjoint-scatter to split
        let data: Vec<i32> = (0..200000).collect();
        let src = MathStorage::<i32>::from_buf(&<m![Big]>::to_value(), data.clone());
        let out = src.reshape::<m![Big], m![Big]>();
        assert_eq!(out.into_buf(&<m![Big]>::to_value()), data);
    }

    /// `from_opt_buf` -> `to_buf_opt` round-trips a PADDED wire buffer: padding positions carry no
    /// canonical slot, so they must read back `Opt::Uninit`, and the live positions must survive in wire
    /// order. Pins the `scatter_buf` (write) and `to_buf_opt` (read) parallel wire walks on padding.
    /// Sized so the wire buffer (`H * 2 * E`) exceeds `PAR_MIN_JOB`, forcing both wire walks to split.
    #[test]
    fn opt_buf_round_trip_preserves_padding() {
        // H * (1#2) * E = 20000 * 2 * 2 = 80000 > PAR_MIN_JOB (65536): the wire walk splits multi-job.
        axes![H = 20000, E = 2];
        // m![H, 1 # 2, E]: the live half is I=0 (E present), the I=1 half is padding.
        let map = <m![H, 1 # 2, E]>::to_value();
        let wire: Vec<Opt<i32>> = (0..map.size())
            .map(|p| if p % 4 < 2 { Opt::Init(p as i32) } else { Opt::Uninit })
            .collect();
        let storage = MathStorage::<i32>::from_opt_buf(&map, wire.clone());
        assert_eq!(storage.to_buf_opt(&map), wire, "padded Opt round-trip must be identity");
    }

    /// `uninit` yields a dense-sized, all-`Uninit` buffer even at a size that forces the parallel fill
    /// to split (> PAR_MIN_JOB), pinning `par_uninit_vec`'s split branch.
    #[test]
    fn uninit_is_dense_and_all_uninit() {
        axes![Big = 100000]; // > PAR_MIN_JOB forces par_uninit_vec's parallel branch
        let s = MathStorage::<i32>::uninit(&<m![Big]>::to_value());
        assert_eq!(s.to_buf_opt(&<m![Big]>::to_value()).len(), 100000);
        assert!(
            s.to_buf_opt(&<m![Big]>::to_value())
                .iter()
                .all(|c| matches!(c, Opt::Uninit)),
            "uninit must be all Opt::Uninit"
        );
    }

    /// `par_uninit_vec` returns exactly `size` all-`Uninit` cells across both branches (below and above
    /// the `PAR_MIN_JOB` serial-floor threshold).
    #[test]
    fn par_uninit_vec_both_branches() {
        for size in [
            0usize,
            7,
            super::PAR_MIN_JOB - 1,
            super::PAR_MIN_JOB,
            super::PAR_MIN_JOB + 1234,
        ] {
            let v = par_uninit_vec::<i32>(size);
            assert_eq!(v.len(), size);
            assert!(v.iter().all(|c| matches!(c, Opt::Uninit)), "size {size} not all Uninit");
        }
    }

    /// A re-FACTORING reshape sized `> PAR_MIN_JOB` so the disjoint scatter splits across workers, with
    /// a genuinely different destination axis structure (`[Big] -> [Hi, Lo]`, one axis vs two) than the
    /// source. `MathStorage::reshape` reinterprets the shared canonical wire buffer, so a correct
    /// scatter re-lands each live cell at the SAME flat position under the new factoring: reading the
    /// reshaped result back in `[Hi, Lo]` wire order reproduces the source's flat buffer. A scatter that
    /// dropped or mis-placed a live cell across the multi-job split would diverge from this oracle.
    #[test]
    fn reshape_refactor_large_matches_oracle() {
        // Hi * Lo = 300 * 257 = 77100 > PAR_MIN_JOB (65536); the dst factors the flat axis into two.
        axes![Big = 77100];
        axes![Hi = 300, Lo = 257];
        let src_wire: Vec<i32> = (0..77100).collect();
        let src = MathStorage::<i32>::from_buf(&<m![Big]>::to_value(), src_wire.clone());
        let out = src.reshape::<m![Big], m![Hi, Lo]>();
        // The wire buffer is shared and row-major, so `[Hi, Lo]` wire order visits the same flat cells in
        // the same order as the flat `[Big]` axis: the reshaped buffer read back equals the source wire.
        assert_eq!(
            out.clone().into_buf(&<m![Hi, Lo]>::to_value()),
            src_wire,
            "refactoring reshape preserves the wire buffer"
        );
        // And reshaping back to the flat axis round-trips to the input.
        let back = out.reshape::<m![Hi, Lo], m![Big]>();
        assert_eq!(
            back.into_buf(&<m![Big]>::to_value()),
            src_wire,
            "reshape round-trip must be identity"
        );
    }

    /// Direct `transpose` coverage, the two parallel paths it takes. (a) Equal-axes, no offset: the
    /// `par_chunks_mut` fast-copy memcpy; sized `> PAR_MIN_JOB` so it splits. (b) A genuine axis permute
    /// `[A, B] -> [B, A]` through the disjoint scatter, vs a nested-loop oracle, also `> PAR_MIN_JOB`.
    #[test]
    fn transpose_fast_copy_and_permute_large() {
        // (a) Equal-axes fast copy, > PAR_MIN_JOB.
        axes![Big = 100000];
        let data: Vec<i32> = (0..100000).collect();
        let src = MathStorage::<i32>::from_buf(&<m![Big]>::to_value(), data.clone());
        let mut dst = MathStorage::<i32>::uninit(&<m![Big]>::to_value());
        dst.transpose::<m![Big], m![Big]>(&src, &Index::new(), &Index::new(), false);
        assert_eq!(
            dst.into_buf(&<m![Big]>::to_value()),
            data,
            "equal-axes fast copy must be identity"
        );

        // (b) Permuting transpose > PAR_MIN_JOB: out[b, a] = in[a, b].
        axes![A = 257, B = 257];
        let (a_ext, b_ext) = (257usize, 257);
        let in_wire: Vec<i32> = (0..(a_ext * b_ext) as i32).collect();
        let src = MathStorage::<i32>::from_buf(&<m![A, B]>::to_value(), in_wire.clone());
        let mut dst = MathStorage::<i32>::uninit(&<m![B, A]>::to_value());
        dst.transpose::<m![A, B], m![B, A]>(&src, &Index::new(), &Index::new(), false);
        let got = dst.into_buf(&<m![B, A]>::to_value());
        let mut expected = vec![0i32; a_ext * b_ext];
        for a in 0..a_ext {
            for b in 0..b_ext {
                expected[b * a_ext + a] = in_wire[a * b_ext + b];
            }
        }
        assert_eq!(got, expected, "permuting transpose must match the nested-loop oracle");
    }

    /// The slow per-position projection: `index(p) + offset`, `finalize`d to coordinates and flattened
    /// by the canonical per-axis weights, the exact definition the lazy iterator must reproduce.
    /// `None` at a pad position.
    fn slow_offsets(mapping: &Mapping, axes: &[AxisTerm], offset: &Index) -> Vec<Option<usize>> {
        let mut strides = vec![1usize; axes.len()];
        for k in (0..axes.len().saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * axes[k + 1].modulo;
        }
        mapping
            .indexes()
            .into_iter()
            .map(|mut index| {
                index.add(offset.clone());
                crate::storage::axis_coords(axes, index)
                    .map(|coords| coords.iter().zip(&strides).map(|(c, s)| c * s).sum())
            })
            .collect()
    }

    /// The offset iteration reproduces the exact per-position projection, bit-for-bit.
    fn assert_iter_matches(mapping: &Mapping, offset: &Index) {
        let axes = mapping.axes();
        let walked: Vec<Option<usize>> = mapping.iter(&axes, offset, true).collect();
        assert_eq!(
            walked,
            slow_offsets(mapping, &axes, offset),
            "iteration disagrees with the slow path\n  mapping={mapping:?}\n  offset={offset:?}"
        );
    }

    /// Plain symbol mappings (the arithmetic-`delta` path). Unshifted full mappings, plus a resized
    /// window with an in-bounds shift, the real view pattern (`tile`/`slice` resize the axis first,
    /// so the shifted window stays within the parent extent). See the caveat on
    /// `mapping_iter`: an out-of-range shift (full axis shifted past its end) is the one input
    /// where the arithmetic path wraps rather than pads, but valid views never build one.
    #[test]
    fn symbol_iter_matches_slow_path() {
        axes![P = 4, Q = 4];
        assert_iter_matches(&<m![P, Q]>::to_value(), &Index::new());
        assert_iter_matches(&<m![Q, P]>::to_value(), &Index::new());
        // A 2-wide window of Q shifted by 1 (reads Q ∈ {1, 2}, within Q's extent 4).
        let mut shift = Index::new();
        shift.add_mapping::<m![Q]>(1);
        assert_iter_matches(&<m![P, Q = 2 # 4]>::to_value(), &shift);
    }

    /// Composite mapping and composite *shift* (the unified hybrid the old dense fallback special-cased).
    /// `[A, B] # 64` packs `A`,`B` into one composite segment, so the iterator builds a composite table; a
    /// shift built from the same mapping carries a composite term that must fold into that table's
    /// `finalize`. Every live shift must reproduce the exact projection.
    #[test]
    fn composite_shift_matches_slow_path() {
        axes![A = 3, B = 5];
        let mapping = <m![[A, B] # 64]>::to_value();
        assert_iter_matches(&mapping, &Index::new());
        let live: usize = mapping.axes().iter().map(|t| t.modulo).product();
        for v in 0..live {
            let mut shift = Index::new();
            shift.add_mapping::<m![[A, B] # 64]>(v);
            assert_iter_matches(&mapping, &shift);
        }
    }

    /// The bounded sub-walk (`MappingIter::range`, the unit a parallel driver splits into) agrees with
    /// the forward walk: every split point's two pieces concatenate to the whole (the seek resumes mid
    /// walk), and `next_back` reverses it. A sub-`PAR_MIN_JOB` op never actually splits, so without this
    /// the seek and double-ended paths go unexercised.
    fn assert_range_matches(mapping: &Mapping, padding: bool) {
        let axes = mapping.axes();
        let full: Vec<Option<usize>> = mapping.iter(&axes, &Index::new(), padding).collect();
        let base = mapping.iter(&axes, &Index::new(), padding);
        let n = base.len();
        assert_eq!(n, full.len(), "len disagrees\n  mapping={mapping:?}");
        for split in [0, 1, n / 3, n / 2, n] {
            let mid = split.min(n);
            let mut concat: Vec<Option<usize>> = base.range(0, mid).collect();
            concat.extend(base.range(mid, n));
            assert_eq!(concat, full, "range split at {mid}\n  mapping={mapping:?}");
        }
        let mut reversed = full.clone();
        reversed.reverse();
        assert_eq!(
            base.range(0, n).rev().collect::<Vec<_>>(),
            reversed,
            "next_back disagrees\n  mapping={mapping:?}"
        );
    }

    /// Range / `next_back` equivalence across the fast path (symbol), the plain odometer (composite),
    /// and padding — the cases a parallel split must reproduce.
    #[test]
    fn range_and_back_match_forward() {
        axes![P = 4, Q = 4, A = 3, B = 5];
        assert_range_matches(&<m![P]>::to_value(), true); // single-factor fast path (row == whole walk)
        assert_range_matches(&<m![P, Q]>::to_value(), true); // multi-factor fast path
        assert_range_matches(&<m![Q, P]>::to_value(), false);
        assert_range_matches(&<m![[A, B] # 64]>::to_value(), true); // composite: plain odometer
        assert_range_matches(&<m![P, Q = 2 # 4]>::to_value(), true); // padded
    }
}
