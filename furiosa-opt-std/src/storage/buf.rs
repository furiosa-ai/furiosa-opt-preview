use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::backend::op_prep::{broadcast_axes, gather_params, scatter_params, transpose_broadcast};
use crate::scalar::*;

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
    pub(crate) fn map<D2: Scalar>(&self, f: impl FnMut(D) -> D2) -> BufStorage<D2> {
        BufStorage::from_vec(self.data.iter().copied().map(f).collect())
    }

    /// Element-wise zip of two same-layout physical buffers, bare `D`, offset-aligned. Backs
    /// [`crate::backend::Backend::zip_with`].
    pub(crate) fn zip_with<D2: Scalar, D3: Scalar>(
        &self,
        other: &BufStorage<D2>,
        mut f: impl FnMut(D, D2) -> D3,
    ) -> BufStorage<D3> {
        let data = self
            .data
            .iter()
            .copied()
            .zip(other.data.iter().copied())
            .map(|(x, y)| f(x, y))
            .collect();
        BufStorage::from_vec(data)
    }

    /// Element-wise ternary zip over the physical buffer. Ternary peer of [`Self::zip_with`].
    pub(crate) fn zip3_with<D2: Scalar, D3: Scalar, D4: Scalar>(
        &self,
        b: &BufStorage<D2>,
        c: &BufStorage<D3>,
        f: impl Fn(D, D2, D3) -> D4,
    ) -> BufStorage<D4> {
        let data = self
            .data
            .iter()
            .zip(&b.data)
            .zip(&c.data)
            .map(|((&x, &y), &z)| f(x, y, z))
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
        // don't-care, which `live_mask` then gates out of the write below.
        let config = &sequence(&[src_view], &[dst_view], SequencerMode::Carve)
            .expect("transpose: Src must be covered by the dst stream")[0];
        // The src window's physical base, and the dst window's base plus the dst liveness mask. Both
        // bases default to 0 for an empty (whole-tensor) offset; the dst mask gates every write so a
        // padding / dead cell is skipped (see `live_mask`). The base resolves the offset against the
        // *base* (live-axis) map, where a tiled offset axis the view carries as padding is still a
        // live `Symbol` (the view shares the base's wire layout, so the wire position carries over).
        let src_base = window_base(src_map, src_offset);
        let dst_base = window_base(dst_map, dst_offset);
        let dst_live = live_mask(dst_view);
        for (dst_pos, src_pos) in config.buffer_offsets().enumerate() {
            if dst_live[dst_pos] {
                self.data[dst_base + dst_pos] = src.data[src_base + src_pos];
            }
        }
    }

    /// Reduces the factors of `self`'s mapping that are absent in `Dst`. `Dst` must be a factor of
    /// the source mapping (only reduce away existing axes; no broadcast). Backs [`crate::backend::Backend::reduce`].
    pub(crate) fn reduce<Src: M, Dst: M, R: Fn(D, D) -> D>(
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
        // Append the broadcast axes (new in Dst) to the Src stream as the inner run; memory = Dst, so
        // each (Src cell, broadcast position) maps to its Dst cell. Empty broadcast ⇒ plain reduce
        // (`bsize == 1`, the stream is just Src).
        let bsize = broadcast.size();
        let stream = src.pair(broadcast);
        let config = &sequence(&[&dst], &[&stream], SequencerMode::Read)
            .expect("reduce: Dst must be Src's kept axes plus the broadcast axes")[0];
        let mut acc: Vec<D> = vec![identity; dst.size()];
        for (stream_pos, dst_cell) in config.buffer_offsets().enumerate() {
            acc[dst_cell] = reduce_fn(acc[dst_cell], self.data[stream_pos / bsize]);
        }
        Self::from_vec(acc)
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
            .buffer_offsets()
            .collect();
        let dst_off: Vec<usize> = sequence(&[&Dst::to_value()], &[&payload.pair(scatter_axis)], SequencerMode::Read)
            .expect("scatter: Dst factors into payload x scatter-axis")[0]
            .buffer_offsets()
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
            .buffer_offsets()
            .collect();
        let dst_off: Vec<usize> = sequence(&[&Dst::to_value()], &[&payload.pair(idx_residue)], SequencerMode::Read)
            .expect("gather: Dst factors into payload x idx-residue")[0]
            .buffer_offsets()
            .collect();
        for payload_pos in 0..dst_off.len() / residue_size {
            for r in 0..residue_size {
                let src_cell = src_off[payload_pos * axis_size + indices[r]];
                let dst_cell = dst_off[payload_pos * residue_size + r];
                dst.data[dst_cell] = self.data[src_cell];
            }
        }
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
            .iter(&src_map.axes(), &Index::new(), true)
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
    for (cell, offset) in data.iter_mut().zip(dst_map.iter(&dst_map.axes(), &Index::new(), true)) {
        if offset.is_some() {
            *cell = live
                .next()
                .expect("transmute: dst_map has more live cells than src_map");
        }
    }
    data
}

/// A per-wire-position liveness mask: `false` where the cell is padding. Lets the relayout no-op on
/// padding, mirroring `MathStorage`'s `None`-gated writes. The lazy wire walk (`Some` = live) replaces
/// a per-position `finalize`.
fn live_mask(mapping: &MappingValue) -> Vec<bool> {
    mapping
        .iter(&mapping.axes(), &Index::new(), true)
        .map(|offset| offset.is_some())
        .collect()
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
