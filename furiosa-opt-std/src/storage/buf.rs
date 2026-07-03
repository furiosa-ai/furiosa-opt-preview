use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;
use rayon::prelude::*;

use crate::backend::op_prep::{broadcast_axes, gather_params, scatter_params, transpose_broadcast};
use crate::cast::{Cast, ContractionCast};
use crate::scalar::*;
use crate::storage::par_iters::{MappingCells, ParMappingExt, ParSequencerExt};
use crate::storage::{DisjointScatter, PAR_MIN_JOB, min_cells_per_job};

/// Emulation / Npu tensor: a bare flat `Vec<D>` physical staging buffer of length `Mapping::SIZE`.
/// `data[i]` is buffer cell `i`, padding cells included and zero-initialized.
///
/// The buffer carries no layout of its own; ops get the mapping(s) from their type parameters and
/// drive the buffer with them. Element-wise ops walk cells by offset; restructuring ops
/// (transpose / reduce / scatter / gather) build a sequencer config from the mappings and walk
/// that (the offset-addressed, finalize-free path).
//
// Why `Vec<D>` and not `ArrayVec<D, Mapping::SIZE>`: keying capacity on the associated const
// `Mapping::SIZE` needs `generic_const_exprs` (unstable/incomplete) and would spread
// `[(); Mapping::SIZE]:` bounds across every method; and stack-inlining a tensor-sized array is
// wrong for large mappings. A heap `Vec` is the right fit for a variable-length staging buffer.
#[derive(Clone, Debug)]
pub struct BufStorage<D: Scalar> {
    pub(crate) data: Vec<D>,
}

impl<D: Scalar> PartialEq for BufStorage<D> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl<D: Scalar + Eq> Eq for BufStorage<D> {}

impl<D: Scalar> BufStorage<D> {
    fn from_vec(data: Vec<D>) -> Self {
        Self { data }
    }

    /// Element-wise map to a new scalar, `BufStorage`'s strategy: walk the physical buffer by offset
    /// over bare `D` (a staging cell is always a defined value; no `Opt`). Backs [`crate::backend::Backend::map`].
    pub(crate) fn map<D2: Scalar>(&self, f: impl Fn(D) -> D2 + Sync) -> BufStorage<D2> {
        // Cells are independent; map them across the rayon pool (the elementwise peer of the buf
        // transpose/reduce parallelism). rayon's adaptive split handles small/large alike.
        BufStorage::from_vec(self.data.par_iter().map(|&x| f(x)).collect())
    }

    /// Element-wise zip of two same-layout physical buffers, bare `D`, offset-aligned. Backs
    /// [`crate::backend::Backend::zip_with`].
    pub(crate) fn zip_with<D2: Scalar, D3: Scalar>(
        &self,
        other: &BufStorage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> BufStorage<D3> {
        let data = (self.data.as_slice(), other.data.as_slice())
            .into_par_iter()
            .map(|(&x, &y)| f(x, y))
            .collect();
        BufStorage::from_vec(data)
    }

    /// Element-wise ternary zip over the physical buffer. Ternary peer of [`Self::zip_with`].
    pub(crate) fn zip3_with<D2: Scalar, D3: Scalar, D4: Scalar>(
        &self,
        b: &BufStorage<D2>,
        c: &BufStorage<D3>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> BufStorage<D4> {
        let data = (self.data.as_slice(), b.data.as_slice(), c.data.as_slice())
            .into_par_iter()
            .map(|(&x, &y, &z)| f(x, y, z))
            .collect();
        BufStorage::from_vec(data)
    }

    /// Writes a transposed/broadcast view of `src` into `self` (the destination) via a sequencer
    /// walk over the physical buffer. `src_map` / `dst_map` are the two storages' base (live-axis)
    /// mappings, used only to resolve a partial-view offset's wire base (see [`window_base`]). Backs
    /// [`crate::backend::Backend::transpose`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transpose<Src: M, Mapping: M>(
        &mut self,
        src: &BufStorage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &MappingValue,
        dst_map: &MappingValue,
        allow_broadcast: bool,
    ) {
        // Structural check (also asserts `Src` is contained in `Mapping`); `!allow_broadcast`
        // rejects a non-padding leftover.
        let _ = transpose_broadcast::<Src, Mapping>(allow_broadcast);
        let src_view = &Src::to_value();
        let dst_view = &Mapping::to_value();
        // Relayout: sequence the `Src` buffer (memory) against the `Mapping` (dst) layout (stream)
        // under `Carve`. Each dst buffer position reads one Src cell; a broadcast axis (in dst, not Src)
        // gets `memory_stride` 0, so one Src cell feeds the whole broadcast run. `Carve` (not `Read`)
        // tolerates a `Bottom` pad in the dst stream, a `view_mut().tile()` write hole, read as a `Top`
        // don't-care, which the dst liveness walk then gates out of the write below.
        let config = &sequence(&[src_view], &[dst_view], SequencerMode::Carve)
            .expect("transpose: Src must be covered by the dst stream")[0];
        // The src window's physical base and the dst window's base. Both default to 0 for an empty
        // (whole-tensor) offset. The base resolves the offset against the *base* (live-axis) map, where
        // a tiled offset axis the view carries as padding is still a live `Symbol` (the view shares the
        // base's wire layout, so the wire position carries over).
        let src_base = window_base(src_map, src_offset);
        let dst_base = window_base(dst_map, dst_offset);
        // Walk the dst window in parallel, zipping three position-aligned streams of the dst layout:
        // the dst cells (each position writes its own, `dst_base + pos`), the src offset to read
        // (`config.par_iter()`, splits via seek), and the dst liveness (`par_iter_cells()` yields `None` for a
        // padding / write-hole cell) — so liveness is read inline, not materialized into a mask. The dst
        // slice runs to the buffer end (every *live* position is in bounds); the stream's trailing
        // `Bottom`-pad positions, which would address past the buffer, are dropped by `zip` taking the
        // shorter length (all dead anyway).
        let src_data = &src.data;
        (
            self.data[dst_base..].par_iter_mut(),
            config.par_iter(),
            dst_view.par_iter_cells(),
        )
            .into_par_iter()
            .with_min_len(PAR_MIN_JOB)
            .for_each(|(slot, src_pos, live)| {
                if live.is_some() {
                    *slot = src_data[src_base + src_pos];
                }
            });
    }

    /// Reduces the factors of `self`'s mapping that are absent in `Dst`. `Dst` must be a factor of
    /// the source mapping (only reduce away existing axes; no broadcast). Backs [`crate::backend::Backend::reduce`].
    ///
    /// Per-output-cell, like [`MathStorage::reduce`]: each output cell independently folds its block of
    /// the reduced (residue) axes into its own slot (`par_iter_mut`), no shared accumulator. The output
    /// cell's source read base is the cell's coordinate projected onto the source (a broadcast axis `Dst`
    /// adds projects to 0, so its fan shares one base); the residue deltas are the reduced axes, shared
    /// across cells. So a read is the additive `base + residue_delta` over the physical buffer.
    ///
    /// REASSOCIATES: one output cell's residue block is folded left-to-right in `residue` order, which is
    /// the physical (wire) order of the reduced axes, not necessarily the serial-`iter` stream order, so
    /// `reduce_fn` runs in an implementation-defined order. For an associative `reduce_fn` the result
    /// equals the serial one; for a non-associative one (e.g. `f32` add/mul) the Emulation result can
    /// differ from serial. Accepted: Emulation is not bit-reproducible for non-associative reductions
    /// (Simulation's per-cell fold over its canonical residue is). Unlike the old per-worker fold, the
    /// order no longer varies run-to-run with rayon's split (each cell folds its whole block alone).
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
        // The kept axes (`Dst` minus the broadcast) sit in both source and Dst; the reduced (residue)
        // axes are the source's beyond them. A broadcast axis (in `dst`, not the source) reads nothing,
        // so its fan copies share one source base and recompute the fold.
        let inter = dst.carve(&broadcast);
        let reduce_residue = src.carve(&inter);
        let out_size = dst.size();
        let plan = ReadPlan::new(&src, &dst, &reduce_residue);
        let data = &self.data;
        let mut out = vec![identity; out_size];
        // Output cells are independent, so fold each across the rayon pool; `with_min_len` holds each job
        // at >= `PAR_MIN_JOB` work (cells x live residue) so small reduces stay one job. A dead (padding)
        // cell has no reads and keeps its `identity`. A split-then-padded reduced axis can land a read
        // past the physical buffer, so skip an out-of-range one (mirrors `MathStorage`).
        out.par_iter_mut()
            .with_min_len(min_cells_per_job(plan.cell_work()))
            .enumerate()
            .for_each(|(o, slot)| {
                let Some(reads) = plan.reads(o) else { return };
                let mut acc = identity;
                for p in reads {
                    if let Some(&v) = data.get(p) {
                        acc = reduce_fn(acc, v);
                    }
                }
                *slot = acc;
            });
        Self::from_vec(out)
    }

    /// Fused contraction (generalized matmul) for the physical buffer: each output cell independently
    /// sums `lhs * rhs` over the contracted axes (those in `Union` but absent from `Out`). O(`Out`)
    /// memory, no `Union` outer product. Backs [`crate::backend::Backend::contraction`] for Emulation.
    ///
    /// Per-output-cell, like [`MathStorage::contraction`]: each operand offset splits into the output
    /// cell's read base plus a contracted-coordinate delta, and each output cell folds that block into
    /// its own slot (`par_iter_mut`), no shared accumulator. An operand axis absent from `Out` (a
    /// contracted axis) is a residue delta; an axis the operand lacks broadcasts (its base / delta is 0).
    ///
    /// Accumulates in `<D as ContractionCast>::Output`, like [`MathStorage::contraction`]: each operand
    /// widens on load via [`Cast`] (`D -> Output`), the contracted block sums in the wider type, and the
    /// sum narrows via [`Cast`] (`Output -> D`) into the slot. So an `i8` matmul accumulates in `i32` and
    /// a `bf16` matmul in `f32`, instead of overflowing the narrow dtype. Both backends are pinned to the
    /// same result by the per-backend bf16 test below (`contraction_bf16_recovers_...`); the i8 fold
    /// shares this code and is deferred to dedicated coverage (#176 int-GEMM).
    ///
    /// One `Backend::contraction` is a single within-slice contraction (its `Union`/`Out` carry no
    /// `Slice` axis), so its narrow is the per-slice Lane Folder narrow. A cross-slice reduction is a
    /// separate downstream Vector Engine `reduce` over the already-narrowed per-slice results, so the
    /// narrow here never spans slices (pinned by `cross_slice_reduce_narrows_per_lane_fold_not_globally`).
    ///
    /// REASSOCIATES, like [`Self::reduce`]: one output cell's products are summed in physical (wire)
    /// order of the contracted axes, not the serial-`iter` stream order. Exact for an associative
    /// combine; for `f32` add/mul the Emulation result can differ from serial. Accepted (Simulation's
    /// per-output-cell fold stays bit-identical), but no longer varies run-to-run with rayon's split.
    pub(crate) fn contraction<Union: M, Lhs: M, Rhs: M, Out: M>(
        lhs: &BufStorage<D>,
        rhs: &BufStorage<D>,
    ) -> BufStorage<D>
    where
        D: ContractionCast,
    {
        let union = Union::to_value();
        let lhs_map = Lhs::to_value();
        let rhs_map = Rhs::to_value();
        let out_map = Out::to_value();
        // The contracted axes are `Union` beyond the kept output, exactly `reduce`'s residue.
        let contracted = union.carve(&out_map);
        let out_size = out_map.size();
        // One read base per output cell into each operand (the output coordinate, contracted at 0), plus
        // the per-contracted-coordinate operand offset deltas shared across cells; the residue is the
        // contracted axes (an operand axis absent from `Out` reduces, one it lacks broadcasts).
        let lhs_plan = ReadPlan::new(&lhs_map, &out_map, &contracted);
        let rhs_plan = ReadPlan::new(&rhs_map, &out_map, &contracted);
        let (lhs_data, rhs_data) = (&lhs.data, &rhs.data);
        let mut out = vec![num_traits::Zero::zero(); out_size];
        // Each output cell sums the products over its contracted block (the two plans' reads zip
        // position-aligned) in `<D as ContractionCast>::Output`: operands widen on load via `Cast`
        // (`D -> Output`), the sum narrows via `Cast` (`Output -> D`). A dead (padding) cell has no
        // reads and keeps its zero. A split-then-padded contracted coordinate can land past a buffer,
        // so skip an out-of-range read (mirrors `MathStorage`).
        out.par_iter_mut()
            .with_min_len(min_cells_per_job(lhs_plan.cell_work()))
            .enumerate()
            .for_each(|(o, slot)| {
                let (Some(lhs_reads), Some(rhs_reads)) = (lhs_plan.reads(o), rhs_plan.reads(o)) else {
                    return;
                };
                // Sum the products in `ContractionCast::Output`: each MAC widens both operands via
                // `Cast`, `filter_map` drops a split-then-padded out-of-range read, and the accumulator
                // narrows via `Cast` after the block folds (`fold` seed = wide zero). The narrow sits
                // outside the fold, the Lane Folder's narrow-once-after-the-reduction.
                let acc = lhs_reads
                    .zip(rhs_reads)
                    .filter_map(|(lp, rp)| match (lhs_data.get(lp), rhs_data.get(rp)) {
                        (Some(&l), Some(&r)) => Some(Cast::cast(l) * Cast::cast(r)),
                        _ => None,
                    })
                    .fold(
                        <<D as ContractionCast>::Output as num_traits::Zero>::zero(),
                        |acc, prod| acc + prod,
                    );
                *slot = Cast::cast(acc);
            });
        Self::from_vec(out)
    }

    /// Scatters `self` into `dst` at positions read from the `i32` index tensor. Backs
    /// [`crate::backend::Backend::scatter`].
    pub(crate) fn scatter<Src: M, Key: M, Dst: M, Idx: M>(
        &self,
        dst: &mut BufStorage<D>,
        index: &BufStorage<i32>,
        scaled: bool,
    ) {
        let key = Key::to_value();
        let (payload, dst_term) = scatter_params(&Src::to_value(), &Dst::to_value(), &key);
        let payload = payload.remove_padding();
        let scatter_axis = MappingValue::from_terms(std::iter::once(dst_term.to_term()));
        // The index tensor's buffer holds exactly one cell per `Idx` position, which is why
        // `decode_indices` reads it whole.
        debug_assert_eq!(index.data.len(), Idx::SIZE);
        let indices = decode_indices(index, decode_stride::<D>(&payload, scaled));

        let key_size = key.size();
        let axis_size = scatter_axis.size();
        let src_off: Vec<usize> = sequence(&[&Src::to_value()], &[&payload.clone().pair(key)], SequencerMode::Read)
            .expect("scatter: Src factors into payload x key")[0]
            .iter()
            .collect();
        let dst_off: Vec<usize> = sequence(&[&Dst::to_value()], &[&payload.pair(scatter_axis)], SequencerMode::Read)
            .expect("scatter: Dst factors into payload x scatter-axis")[0]
            .iter()
            .collect();
        for payload_pos in 0..src_off.len() / key_size {
            for key_pos in 0..key_size {
                let src_cell = src_off[payload_pos * key_size + key_pos];
                let dst_cell = dst_off[payload_pos * axis_size + indices[key_pos]];
                dst.data[dst_cell] = self.data[src_cell];
            }
        }
    }

    /// Gathers from `self` (table) into `dst` at positions read from the index tensor. Backs
    /// [`crate::backend::Backend::gather`].
    pub(crate) fn gather<Src: M, Dst: M, Idx: M>(
        &self,
        dst: &mut BufStorage<D>,
        index: &BufStorage<i32>,
        scaled: bool,
    ) {
        let params = gather_params(&Src::to_value(), &Dst::to_value(), &Idx::to_value());
        let payload = params.payload.remove_padding();
        let gather_axis = MappingValue::from_terms(std::iter::once(params.src_term.to_term()));
        // The compact residue axes are exactly the index tensor's mapping.
        let idx_residue = Idx::to_value();
        let indices = decode_indices(index, decode_stride::<D>(&payload, scaled));

        let axis_size = gather_axis.size();
        let residue_size = idx_residue.size();
        let src_off: Vec<usize> = sequence(
            &[&Src::to_value()],
            &[&payload.clone().pair(gather_axis)],
            SequencerMode::Read,
        )
        .expect("gather: Src table factors into payload x gather-axis")[0]
            .iter()
            .collect();
        let dst_off: Vec<usize> = sequence(&[&Dst::to_value()], &[&payload.pair(idx_residue)], SequencerMode::Read)
            .expect("gather: Dst factors into payload x idx-residue")[0]
            .iter()
            .collect();
        // Each payload block writes its `residue_size` dst cells, and `dst_off` is a permutation of dst
        // positions (distinct), so the blocks scatter into disjoint slots and parallelize. The index
        // only repeats *reads* (`src_off[... + indices[r]]`), never a write. Same raw-pointer disjoint
        // scatter `transpose` uses (safe Rust cannot alias disjoint indices of one `&mut [_]`).
        let src_data = &self.data;
        // Each payload step scatters an inner run of `residue_size` cells, so this keeps the explicit
        // rayon loop (not one write per step); the scatter itself goes through the shared bounds-checked
        // `DisjointScatter`, which fails loud on an out-of-range `dst_cell` rather than corrupting memory.
        let scatter = DisjointScatter::new(&mut dst.data);
        let scatter = &scatter;
        (0..dst_off.len() / residue_size)
            .into_par_iter()
            .with_min_len(min_cells_per_job(residue_size))
            .for_each(|payload_pos| {
                for r in 0..residue_size {
                    let src_cell = src_off[payload_pos * axis_size + indices[r]];
                    let dst_cell = dst_off[payload_pos * residue_size + r];
                    // SAFETY: `dst_off` is a permutation of dst positions, so every `dst_cell` is
                    // distinct across (payload_pos, r) -- no two parallel jobs touch the same slot.
                    // `DisjointScatter::write` bounds `dst_cell`.
                    unsafe { scatter.write(dst_cell, src_data[src_cell]) };
                }
            });
    }

    /// Reinterprets the physical buffer from `Mapping`-shaped to `Mapping2`-shaped, returning the
    /// result. Backs [`crate::backend::Backend::reshape`].
    pub(crate) fn reshape<Mapping: M, Mapping2: M>(&self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        // same physical buffer; only the type-level mapping changes.
        Self::from_vec(self.data.clone())
    }

    /// Relabels the physical buffer from `src_map` to `dst_map`, compacting away any padding the
    /// relabel drops, so later offset-addressed ops see `dst_map`'s layout. Backs
    /// [`crate::backend::Backend::transmute`]. The vector-engine relayout is
    /// `…tile(k).read().transmute()`: the read leaves the buffer padded in `src_map`, the transmute
    /// narrows it to the compact `dst_map`. Reproduces what canonical (padding-free) `MathStorage`
    /// yields for free.
    pub(crate) fn transmute(self, src_map: &MappingValue, dst_map: &MappingValue) -> Self {
        // Same layout drops no padding.
        if src_map == dst_map {
            return self;
        }
        Self::from_vec(place_live_cells(dst_map, self.live_cells_in_wire_order(src_map)))
    }

    /// The source's live cells in wire order, the real data, padding excluded.
    fn live_cells_in_wire_order(self, src_map: &MappingValue) -> impl Iterator<Item = D> {
        src_map
            .iter_cells()
            .zip(self.data)
            .filter_map(|(offset, cell)| offset.is_some().then_some(cell))
    }

    /// Build from a flat `D` buffer in `mapping`-order.
    pub(crate) fn from_buf(mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self {
        let data: Vec<D> = data.into_iter().collect();
        assert_eq!(mapping.size(), data.len(), "shape mismatch");
        Self::from_vec(data)
    }

    /// Serialize to a flat `D` buffer in `mapping`-order. The storage already IS the physical / wire
    /// buffer, so this moves its `Vec` out (no clone) and the mapping is unused.
    pub(crate) fn into_buf(self, _mapping: &MappingValue) -> Vec<D> {
        self.data
    }
}

/// The per-output-cell read plan for `reduce` / `contraction` over the physical buffer: how to find,
/// for each output cell, the operand buffer positions whose values fold into it. The physical (wire)
/// peer of [`MathStorage`]'s `canonical_bases` + `live_offsets`.
///
/// Built from one sequencer config over the `out`-outer / `residue`-inner stream (which fully covers the
/// operand: `out` carries the kept axes, `residue` the reduced ones; a broadcast axis the operand lacks
/// contributes nothing). The operand wire offset is additive in the two coordinates
/// (`offset(o, r) = base(o) + delta(r)`), the same model the serial per-cell fold uses. The shared
/// `delta(r)` (one residue row, small) is precomputed; each cell's `base(o)` is seeked lazily by
/// [`Self::reads`], so there is no `out_size` base table and the fold is a single parallel pass (the
/// seek that would fill such a table happens inside it instead). [`Self::reads`] `.get()`-guards an
/// out-of-range sum (a split-then-padded coordinate).
struct ReadPlan {
    /// The operand wire offsets over the `out`-outer / `residue`-inner stream; `base(o)` is seeked from it.
    config: SequencerConfig,
    /// Live residue deltas added to a cell's base, in `residue` walk order (padding positions dropped).
    delta: Vec<usize>,
    /// The padded residue width: the stream stride between consecutive output cells.
    rwidth: usize,
    /// The first live residue column within a row (the base read's offset into the row).
    r0: usize,
    /// Total stream length; an output cell whose base position is `>= stream_size` is dead (padding).
    stream_size: usize,
}

impl ReadPlan {
    /// `memory` is the operand mapping, `out` the output mapping (its cell index `o` is the stream row),
    /// `residue` the operand's reduced / contracted axes (the `carve` leftover, which marks the kept /
    /// broadcast axes as `Top` pads).
    fn new(memory: &MappingValue, out: &MappingValue, residue: &MappingValue) -> Self {
        // The residue carries `Top` pads (the kept / broadcast axes `carve` marked, and any padded
        // operand axis): a padded position re-reads its live sibling (`memory_stride` 0), so folding it
        // would double-count. Walk the residue once to mask the live positions (mirrors `MathStorage`'s
        // padding-skipping `live_offsets`); only those become deltas.
        let live_r: Vec<usize> = residue
            .iter_cells()
            .enumerate()
            .filter_map(|(r, off)| off.map(|_| r))
            .collect();
        let rwidth = residue.size(); // the padded residue width: one full stream row below
        // A reduce / contraction always folds at least one cell (an empty residue is degenerate), so the
        // residue has a first live position; index it after this guard rather than panicking opaquely.
        let &r0 = live_r
            .first()
            .expect("reduce/contraction: residue must have a live cell to fold");
        // Operand wire offsets over the `out` x `residue` stream (`out` outer, residue inner). `out`
        // carries the kept axes (a broadcast axis the operand lacks reads stride 0, so its fan shares one
        // base) and `residue` the reduced axes, so together they consume the operand. `out` sequenced as
        // the outer stream means stream row `o` IS output-buffer offset `o`: `offset(o, r)` sits at stream
        // position `o * rwidth + r`.
        let config = sequence(&[memory], &[&out.clone().pair(residue.clone())], SequencerMode::Read)
            .expect("reduce/contraction: out x residue must factor the operand")
            .swap_remove(0);
        let stream_size = config.stream_size();
        // Deltas: the live residue cells of the first reachable output cell's row, each minus that row's
        // first live read. The first reachable cell is `o == 0` when `r0` is in range (the base position
        // `o * rwidth + r0` grows with `o`); otherwise nothing is reachable and the deltas are unused.
        let delta: Vec<usize> = if r0 < stream_size {
            let row: Vec<usize> = config.iter_range(0, rwidth.min(stream_size)).collect();
            live_r.iter().map(|&r| row[r] - row[r0]).collect()
        } else {
            Vec::new()
        };
        Self {
            config,
            delta,
            rwidth,
            r0,
            stream_size,
        }
    }

    /// The operand buffer positions folding into output cell `o`: its base (seeked from the stream) plus
    /// each live residue delta. `None` for a dead (padding) output cell the stream never reaches. A
    /// split-then-padded residue can still push an individual read past the buffer, so the caller
    /// `.get()`-guards each position.
    fn reads(&self, o: usize) -> Option<impl Iterator<Item = usize> + '_> {
        let pos = o * self.rwidth + self.r0;
        let base = (pos < self.stream_size).then(|| self.config.iter_range(pos, pos + 1).next())??;
        Some(self.delta.iter().map(move |&d| base + d))
    }

    /// Work per output cell (the live residue width), for the rayon `with_min_len` job-size hint.
    fn cell_work(&self) -> usize {
        self.delta.len()
    }
}

/// The physical wire base where the `offset` window starts in `base_map` (0 for an empty offset): the
/// wire position whose canonical offset equals the offset's. `base_map` is the storage's live-axis
/// mapping, so a tiled offset axis the relayout's `Src` / `Dst` *type* carries as padding is still a
/// live `Symbol` here and matches; the view shares the base's wire layout, so this position is also the
/// partial view's start in the buffer. Mirrors how `MathStorage::transpose` resolves the same offset
/// against its storage's live `axes`.
fn window_base(base_map: &MappingValue, offset: &Index) -> usize {
    if *offset == Index::new() {
        return 0;
    }
    // The offset's dense (canonical) offset under `base_map`: each axis's coordinate times its
    // canonical weight (innermost axis = 1). One `finalize` (inside `axis_coords`).
    let axes = base_map.axes();
    let coords = crate::storage::axis_coords(&axes, offset.clone())
        .expect("transpose: partial-view offset must be a live coordinate of its base mapping");
    let mut weight = 1;
    let mut target = 0;
    for (axis, &coord) in axes.iter().rev().zip(coords.iter().rev()) {
        target += coord * weight;
        weight *= axis.modulo;
    }
    // The wire position whose canonical offset is `target`, found by the lazy wire walk (`Some` = live,
    // padding never matches), replacing a `finalize` of every position with the cheap odometer.
    base_map
        .iter(&axes, &Index::new(), true)
        .position(|offset| offset == Some(target))
        .expect("transpose: partial-view offset must land on a live cell of its base mapping")
}

/// Lays `live` cells into a fresh `dst_map`-sized buffer at its live wire positions (padding stays
/// zero), mirroring how `MathStorage::into_buf` re-places its canonical cells under `dst_map`.
fn place_live_cells<D: Scalar>(dst_map: &MappingValue, mut live: impl Iterator<Item = D>) -> Vec<D> {
    let mut data = vec![D::zero(); dst_map.size()];
    for (cell, offset) in data.iter_mut().zip(dst_map.iter_cells()) {
        if offset.is_some() {
            *cell = live
                .next()
                .expect("transmute: dst_map has more live cells than src_map");
        }
    }
    data
}

/// Decodes a `BufStorage`-backed `i32` index tensor to element positions: each value divided by the
/// stride from [`decode_stride`] (a byte stride when scaled, else 1). `BufStorage` offsets are
/// physical and `data` already holds exactly the index mapping's cells, so the whole buffer is read.
/// Shared by scatter and gather.
fn decode_indices(index: &BufStorage<i32>, index_stride: usize) -> Vec<usize> {
    index
        .data
        .iter()
        .map(|&v| usize::try_from(v).expect("scatter/gather index must be non-negative") / index_stride)
        .collect()
}

/// The divisor [`decode_indices`] applies: a scaled index tensor holds byte offsets into
/// payload-sized blocks (divide by the block's byte size); an unscaled one holds element indices.
fn decode_stride<D>(payload: &MappingValue, scaled: bool) -> usize {
    if scaled {
        payload.size() * std::mem::size_of::<D>()
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BufStorage::contraction` reproduces a plain matmul. `[M,K] · [K,N] -> [M,N]` with no padding,
    /// so the wire buffers are row-major and the result is hand-checkable.
    #[test]
    fn contraction_matches_matmul() {
        axes![M = 2, K = 3, N = 2];
        // lhs row-major: [[1,2,3],[4,5,6]]; rhs row-major: [[1,2],[3,4],[5,6]].
        let lhs = BufStorage::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = BufStorage::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = BufStorage::contraction::<m![M, K, N], m![M, K], m![K, N], m![M, N]>(&lhs, &rhs);
        // [[1·1+2·3+3·5, 1·2+2·4+3·6], [4·1+5·3+6·5, 4·2+5·4+6·6]] = [[22,28],[49,64]].
        assert_eq!(out.data, vec![22.0f32, 28.0, 49.0, 64.0]);
    }

    /// A `SequencerConfig`'s walk (the parallel-`contraction`/`reduce` offset source): range pieces
    /// concatenate to the full `iter`, and `next_back` reverses it. Guards the seek / double-ended paths.
    #[test]
    fn sequencer_range_and_back_match_full() {
        axes![M = 4, K = 6, N = 4];
        let cfg = sequence(
            &[&<m![M, N]>::to_value()],
            &[&<m![M, K, N]>::to_value()],
            SequencerMode::Read,
        )
        .unwrap();
        let config = &cfg[0];
        let full: Vec<usize> = config.iter().collect();
        let n = config.stream_size();
        assert_eq!(n, full.len());
        for split in [0, 1, n / 2, n] {
            let mid = split.min(n);
            let mut concat: Vec<usize> = config.iter_range(0, mid).collect();
            concat.extend(config.iter_range(mid, n));
            assert_eq!(concat, full, "iter_range split at {mid}");
        }
        let mut reversed = full.clone();
        reversed.reverse();
        assert_eq!(config.iter().rev().collect::<Vec<_>>(), reversed, "next_back");
    }

    /// A reduce big enough (> `PAR_MIN_JOB`) that rayon actually splits the walk across workers. `i32`
    /// add is associative, so the regrouped parallel sum equals the serial one and is deterministic —
    /// the sub-`PAR_MIN_JOB` unit tests never exercise the split itself.
    #[test]
    fn reduce_large_split_matches_serial() {
        axes![R = 256, C = 512]; // 131072 stream positions > PAR_MIN_JOB (65536)
        let data: Vec<i32> = (0..256 * 512).map(|i| i % 5).collect();
        let out = BufStorage::from_vec(data.clone()).reduce::<m![R, C], m![C], _>(|a, b| a + b, 0, false);
        let mut expected = vec![0i32; 512];
        for r in 0..256 {
            for c in 0..512 {
                expected[c] += data[r * 512 + c];
            }
        }
        assert_eq!(out.data, expected);
    }

    /// The `out_size ≫ contracted` corner the per-output-cell rewrite targets: a huge output (> 1M cells)
    /// reduced over a tiny (4-cell) axis. The old per-worker-accumulator fold allocated and combined a
    /// `vec![identity; out_size]` per leaf here, `O(out_size × leaves)` — far more than the real work.
    /// The per-cell fold writes each output cell once. Pins that the unified path stays correct (and the
    /// bench `bench/storage.rs::reduce_wide_thin` pins it is not slower than serial).
    #[test]
    fn reduce_wide_out_thin_contracted_matches_serial() {
        axes![Big = 1048576, Small = 4]; // out = 2^20 cells, contracted = 4
        let data: Vec<i32> = (0..(1 << 20) * 4).map(|i| i % 7).collect();
        let out = BufStorage::from_vec(data.clone()).reduce::<m![Big, Small], m![Big], _>(|a, b| a + b, 0, false);
        let mut expected = vec![0i32; 1 << 20];
        for big in 0..(1 << 20) {
            for small in 0..4 {
                expected[big] += data[big * 4 + small];
            }
        }
        assert_eq!(out.data, expected);
    }

    /// A contraction whose output-cell count comfortably exceeds the per-job floor, so rayon splits the
    /// per-cell fold across workers (the floor is `min_cells_per_job(K=4) = PAR_MIN_JOB/4 = 16384`, so
    /// `M*N = 128*512 = 65536` cells give ~4 jobs — not a single serial run). `i32` products / sums are
    /// associative, so the per-cell result equals the serial reference exactly.
    #[test]
    fn contraction_large_split_matches_serial() {
        axes![M = 128, K = 4, N = 512]; // 65536 output cells >> 16384 floor
        let lhs: Vec<i32> = (0..128 * 4).map(|i| i % 5).collect();
        let rhs: Vec<i32> = (0..4 * 512).map(|i| i % 3).collect();
        let out = BufStorage::contraction::<m![M, K, N], m![M, K], m![K, N], m![M, N]>(
            &BufStorage::from_vec(lhs.clone()),
            &BufStorage::from_vec(rhs.clone()),
        );
        let mut expected = vec![0i32; 128 * 512];
        for m in 0..128 {
            for n in 0..512 {
                for k in 0..4 {
                    expected[m * 512 + n] += lhs[m * 4 + k] * rhs[k * 512 + n];
                }
            }
        }
        assert_eq!(out.data, expected);
    }

    /// `BufStorage::contraction` with a REORDERED output (`Out = m![N, M]`, not the canonical `[M, N]`):
    /// the per-cell `ReadPlan` indexes the base table by `Out`'s own buffer order, so the result must
    /// read back in `[N, M]` order. Guards that `ReadPlan` does not assume canonical out order (the
    /// `BufStorage` peer of `MathStorage`'s `contraction_reordered_out_axes`, which does not transfer
    /// because `BufStorage::contraction` carries explicit `Lhs`/`Rhs`/`Out` params and its own plan).
    #[test]
    fn contraction_reordered_out_matches_serial() {
        axes![M = 2, K = 3, N = 2];
        let lhs = BufStorage::from_vec(vec![1i32, 2, 3, 4, 5, 6]); // [M,K] row-major
        let rhs = BufStorage::from_vec(vec![1i32, 2, 3, 4, 5, 6]); // [K,N] row-major
        let out = BufStorage::contraction::<m![M, K, N], m![M, K], m![K, N], m![N, M]>(&lhs, &rhs);
        // The (M,N) product [[22,28],[49,64]] read back N-major: [22,49,28,64].
        assert_eq!(out.data, vec![22i32, 49, 28, 64]);
    }

    /// `BufStorage::contraction` with a PADDED operand axis (`K = 2 # 4`: K live 2, wire extent 4). The
    /// contracted residue carries the `# 4` Top pad; the live-residue mask must fold only the 2 live K
    /// cells (the padded positions, if folded, would read stale wire cells). Pins the padding guard on
    /// the contraction path (the reduce peer is `reduce_padded_source_math_and_buf`).
    #[test]
    fn contraction_padded_contracted_axis_matches_serial() {
        axes![M = 2, K = 2, N = 2];
        // lhs m![M, K = 2 # 4]: each M row is [k0, k1, pad, pad] in wire order (size 2*4 = 8).
        let lhs = BufStorage::from_vec(vec![1i32, 2, 0, 0, 3, 4, 0, 0]);
        // rhs m![K = 2 # 4, N]: each of the 4 K wire rows holds [n0, n1]; rows 2,3 are pad (size 4*2 = 8).
        let rhs = BufStorage::from_vec(vec![1i32, 2, 3, 4, 0, 0, 0, 0]);
        let out =
            BufStorage::contraction::<m![M, K = 2 # 4, N], m![M, K = 2 # 4], m![K = 2 # 4, N], m![M, N]>(&lhs, &rhs);
        // Only the 2 live K cells fold: out[m,n] = sum_{k<2} lhs[m,k] * rhs[k,n].
        // [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7,10],[15,22]].
        assert_eq!(out.data, vec![7i32, 10, 15, 22]);
    }

    /// `BufStorage::contraction` with a BROADCAST operand: `rhs` (`m![N]`) lacks `M`, so it broadcasts
    /// across the M output rows (its base / M-stride is 0, one rhs cell feeds every M). Pins the
    /// broadcast branch described in the contraction doc (an axis the operand lacks).
    #[test]
    fn contraction_broadcast_operand_matches_serial() {
        axes![M = 2, K = 3, N = 2];
        let lhs_data = vec![1i32, 2, 3, 4, 5, 6]; // [M,K] row-major
        let rhs_data = vec![1i32, 2, 3, 4, 5, 6]; // [K,N] row-major, broadcasts over M
        let out = BufStorage::contraction::<m![M, K, N], m![M, K], m![K, N], m![M, N]>(
            &BufStorage::from_vec(lhs_data.clone()),
            &BufStorage::from_vec(rhs_data.clone()),
        );
        let mut expected = vec![0i32; 4];
        for m in 0..2 {
            for n in 0..2 {
                for k in 0..3 {
                    expected[m * 2 + n] += lhs_data[m * 3 + k] * rhs_data[k * 2 + n];
                }
            }
        }
        assert_eq!(out.data, expected);
    }

    /// A `bf16` `K = 256` contraction accumulates in `f32`, recovering small addends a narrow per-step
    /// accumulator drops: `lhs = [1.0, 1/128, .., 1/128]` against `rhs = ones` (both exact in bf16, so the
    /// only error is accumulation rounding) sums to `1.0 + 255*(1/128)`. A bf16 running sum saturates near
    /// `2.0` and swallows every later `1/128`; the `f32` accumulator keeps them. No padding, so the wire
    /// buffers are the raw vecs.
    #[test]
    fn contraction_bf16_recovers_small_addends_via_wide_accumulator() {
        axes![M = 1, K = 256, N = 1];
        let lhs: Vec<bf16> = std::iter::once(1.0)
            .chain(std::iter::repeat_n(1.0 / 128.0, 255))
            .map(bf16::from_f32)
            .collect();
        let rhs = vec![bf16::from_f32(1.0); 256];
        let out = BufStorage::contraction::<m![M, K, N], m![M, K], m![K, N], m![M, N]>(
            &BufStorage::from_vec(lhs),
            &BufStorage::from_vec(rhs),
        );
        let got = out.data[0].to_f32();
        // Exact f32 reference 1.0 + 255*(1/128); bf16 has 7 mantissa bits so 1 ulp = reference * 2^-7.
        let reference = 2.9921875_f32;
        assert!(
            (got - reference).abs() <= reference * f32::exp2(-7.0),
            "f32 accumulator should recover small addends bf16 drops: got={got}"
        );
    }

    /// The integer counterpart, end-to-end through the buffer fold: an `i16` `K = 256` dot of
    /// `1000 * 1000` sums to `256_000_000`, far past `i16`'s range. The `i32` `ContractionCast::Output`
    /// accumulator holds the true sum and wraps `as i16` once at the narrow (a narrow-typed fold would
    /// wrap at every step), so the output equals `256_000_000 as i16`. No padding, so the wire buffers
    /// are the raw vecs.
    #[test]
    fn contraction_i16_accumulates_wide_then_wraps_once() {
        axes![M = 1, K = 256, N = 1];
        let out = BufStorage::contraction::<m![M, K, N], m![M, K], m![K, N], m![M, N]>(
            &BufStorage::from_vec(vec![1000i16; 256]),
            &BufStorage::from_vec(vec![1000i16; 256]),
        );
        assert_eq!(out.data[0], 256_000_000i32 as i16);
    }

    /// The `Output -> D` narrow happens once per within-slice `contraction` (the Lane Folder narrow), so
    /// a cross-slice reduction over the narrowed per-slice partials differs from one global wide fold.
    /// The hardware narrows each per-slice partial at its own Lane Folder and the Vector Engine sums the
    /// narrowed partials; the sim composes that as three per-slice `BufStorage::contraction`s feeding a
    /// downstream `BufStorage::reduce`.
    ///
    /// Each per-slice dot is `[1.0, 2^-8] . [1.0, 1.0] = 1.0 + 2^-8`, which the per-slice narrow rounds
    /// to `1.0` (`2^-8` is half a bf16 ulp at magnitude 1, ties-to-even). The cross-slice `reduce` over
    /// the three `1.0`s gives `3.0`. A global wide fold is the same six MACs as one `K=6` contraction: it
    /// keeps `3.0 + 3·2^-8` in `f32` and narrows once to `bf16(3.015625)`. The two outputs differ, so the
    /// narrow is per slice, not held wide across the cross-slice reduce.
    #[test]
    fn cross_slice_reduce_narrows_per_lane_fold_not_globally() {
        axes![Slice = 3, M = 1, K = 2, N = 1, GlobalK = 6];
        // Per-slice path: each slice is a within-slice contraction that narrows to bf16 at its lane fold.
        let lhs_slice = || BufStorage::from_vec(vec![bf16::from_f32(1.0), bf16::from_f32(2.0f32.powi(-8))]);
        let rhs_slice = || BufStorage::from_vec(vec![bf16::from_f32(1.0); 2]);
        let partial = BufStorage::contraction::<m![M, K, N], m![M, K], m![K, N], m![M, N]>(&lhs_slice(), &rhs_slice());
        assert_eq!(
            partial.data[0].to_f32(),
            1.0,
            "per-slice narrow rounds 1 + 2^-8 to 1.0 in bf16"
        );
        // The Vector Engine sums the three narrowed per-slice partials downstream (a `reduce`).
        let three_narrowed = BufStorage::from_vec(vec![partial.data[0]; 3]);
        let cross_slice = three_narrowed
            .reduce::<m![Slice, M, N], m![M, N], _>(|a, b| a + b, bf16::from_f32(0.0), false)
            .data[0]
            .to_f32();
        assert_eq!(cross_slice, 3.0);
        // One global wide fold: the same six MACs as a single K=6 contraction, narrowing once at the end.
        let global = BufStorage::contraction::<m![M, GlobalK, N], m![M, GlobalK], m![GlobalK, N], m![M, N]>(
            &BufStorage::from_vec(
                [1.0, 2.0f32.powi(-8)]
                    .into_iter()
                    .cycle()
                    .take(6)
                    .map(bf16::from_f32)
                    .collect(),
            ),
            &BufStorage::from_vec(vec![bf16::from_f32(1.0); 6]),
        );
        assert_eq!(
            global.data[0].to_f32(),
            3.015625,
            "global wide fold keeps 3 + 3·2^-8, narrows once"
        );
        assert_ne!(
            cross_slice,
            global.data[0].to_f32(),
            "per-slice Lane Folder narrow must differ from a global narrow across the cross-slice reduce"
        );
    }

    /// f32 reduce over a `> PAR_MIN_JOB` split: `f32` add is non-associative, so the reassociated parallel
    /// fold need only AGREE WITH SERIAL TO TOLERANCE, not bit-exactly (the documented Emulation
    /// non-reproducibility). The tolerance is tight (`1e-4` relative): reassociating 256 well-conditioned
    /// positive adds drifts only ~`n·eps ≈ 1.5e-5`, so `1e-4` passes genuine reassociation yet still
    /// catches a dropped / double-counted term (~0.4%+), which the associative `i32` tests cannot.
    #[test]
    fn reduce_large_split_f32_close_to_serial() {
        axes![R = 256, C = 512];
        let data: Vec<f32> = (0..256 * 512).map(|i| (i % 13) as f32 * 0.5 + 0.25).collect();
        let out = BufStorage::from_vec(data.clone()).reduce::<m![R, C], m![C], _>(|a, b| a + b, 0.0, false);
        // f64-accumulated reference, so the tight f32 bound measures only the parallel fold's drift.
        let mut serial = vec![0.0f64; 512];
        for r in 0..256 {
            for c in 0..512 {
                serial[c] += data[r * 512 + c] as f64;
            }
        }
        for (got, want) in out.data.iter().zip(&serial) {
            assert!(
                (*got as f64 - want).abs() <= 1e-4 * want.abs().max(1.0),
                "got {got}, serial {want}"
            );
        }
    }
}
