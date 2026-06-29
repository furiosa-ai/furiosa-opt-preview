use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;
use ndarray::{ArrayD, IxDyn};

use crate::backend::op_prep::{broadcast_axes, gather_params, scatter_params, transpose_broadcast};
use crate::scalar::*;

/// `mapping`'s live offsets over `axes`, materialized (no shift). Only for the small per-cell index
/// sets (reduce's residue / fan, scatter/gather's key / residue); the large moves stay lazy.
fn live_offsets(mapping: &MappingValue, axes: &[AxisTerm]) -> Vec<usize> {
    mapping.iter(axes, &Index::new(), false).flatten().collect()
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

    /// The dense canonical buffer as a flat slice (innermost axis contiguous). Infallible: every constructor
    /// (`uninit`'s `from_elem`, `scatter_buf`'s `from_shape_vec`, `map`'s `ArrayD::map`) produces a
    /// standard contiguous array and no method takes a non-contiguous view, so `as_slice` is
    /// always `Some`, the `expect` documents that invariant rather than guarding a reachable case.
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
    pub(crate) fn map<D2: Scalar>(&self, mut f: impl FnMut(D) -> D2) -> MathStorage<D2> {
        let data = self.data.map(|c| match c {
            Opt::Init(x) => Opt::Init(f(*x)),
            Opt::Uninit => Opt::Uninit,
        });
        MathStorage::new(self.axes.clone(), data)
    }

    /// Element-wise zip of two same-layout tensors by storage offset; the result is defined only
    /// where both inputs are (`Uninit` propagates). Backs [`crate::backend::Backend::zip_with`].
    pub(crate) fn zip_with<D2: Scalar, D3: Scalar>(
        &self,
        other: &MathStorage<D2>,
        f: impl Fn(D, D2) -> D3,
    ) -> MathStorage<D3> {
        let data: ArrayD<Opt<D3>> = ndarray::Zip::from(&self.data)
            .and(&other.data)
            .map_collect(|x, y| match (x, y) {
                (Opt::Init(x), Opt::Init(y)) => Opt::Init(f(*x, *y)),
                _ => Opt::Uninit,
            });
        MathStorage::new(self.axes.clone(), data)
    }

    /// Element-wise ternary zip; the result cell is defined only where all three inputs are.
    /// Ternary peer of [`Self::zip_with`].
    pub(crate) fn zip3_with<D2: Scalar, D3: Scalar, D4: Scalar>(
        &self,
        b: &MathStorage<D2>,
        c: &MathStorage<D3>,
        f: impl Fn(D, D2, D3) -> D4,
    ) -> MathStorage<D4> {
        let data: ArrayD<Opt<D4>> = ndarray::Zip::from(&self.data)
            .and(&b.data)
            .and(&c.data)
            .map_collect(|x, y, z| match (x, y, z) {
                (Opt::Init(x), Opt::Init(y), Opt::Init(z)) => Opt::Init(f(*x, *y, *z)),
                _ => Opt::Uninit,
            });
        MathStorage::new(self.axes.clone(), data)
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
        // Fast path. Same wire order, no shift: a plain copy (equal axes admit no broadcast).
        if self.axes == src.axes && *src_offset == Index::new() && *dst_offset == Index::new() {
            self.flat_mut().copy_from_slice(src.flat());
            return;
        }
        let dst_mapping = Mapping::to_value();
        let reads = dst_mapping.iter(&src.axes, src_offset, false);
        let writes = dst_mapping.iter(&self.axes, dst_offset, false);
        let src_flat = src.flat();
        let dst_flat = self.flat_mut();
        for (read, write) in reads.zip(writes) {
            let (Some(read), Some(write)) = (read, write) else {
                continue;
            };
            // Gate out-of-range offsets as the old finalize did (it returned None). An invalid relayout
            // (one the engine's topology check rejects a stage later) can project a live cell past the
            // buffer, so skip it rather than panic, letting the real validation error surface instead of
            // an index-out-of-bounds. Valid relayouts never produce out-of-range offsets.
            if let (Some(&cell), Some(slot)) = (src_flat.get(read), dst_flat.get_mut(write)) {
                *slot = cell;
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
        // Split the mapping into the kept axes (`inter`), the axes folded away (`reduce_residue`), and
        // the kept-but-new broadcast axes. A source read is additive over the canonical buffer
        // (`read = kept_read + residue_delta`), a dst write is `kept_write + fan`.
        let inter = dst.carve(&broadcast);
        let reduce_residue = src.carve(&inter);
        let mut out = MathStorage::<D>::uninit(&dst);
        // The kept read/write streams are walked once (one step per output cell), so stay lazy. The
        // residue and fan, reused inside every kept cell, are materialized once.
        let kept_reads = inter.iter(&self.axes, &Index::new(), false);
        let kept_writes = inter.iter(&out.axes, &Index::new(), false);
        let residue = live_offsets(&reduce_residue, &self.axes);
        let fan = live_offsets(&broadcast, &out.axes);
        let src_flat = self.flat();
        let dst_flat = out.flat_mut();
        for (read_base, write_base) in kept_reads.zip(kept_writes) {
            let (Some(read_base), Some(write_base)) = (read_base, write_base) else {
                continue;
            };
            // Fold this kept cell's residue block, then write the result across its broadcast fan.
            // A split-then-padded reduced axis (e.g. `R # 32` reduced over a sub-factor) has fewer
            // live residue cells in a boundary group than the uniform `residue` list assumes; those
            // padding combinations land past the dense buffer, so skip them (they are not real cells).
            let mut acc = Opt::Init(identity);
            for &delta in &residue {
                if let Some(&cell) = src_flat.get(read_base + delta) {
                    acc = acc.zip_map(cell, &reduce_fn);
                }
            }
            for &fan_off in &fan {
                if let Some(slot) = dst_flat.get_mut(write_base + fan_off) {
                    *slot = acc;
                }
            }
        }
        out
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
        self.indexed_move(dst, &params.payload, &read_inner, &write_inner);
    }

    /// Moves cells from `self` to `dst` over the shared `payload`: walk the payload once (its src-read
    /// and dst-write offsets paired), and for each payload cell move every free cell, adding the
    /// per-free-cell `read_inner` / `write_inner` offsets. The indexed side's offsets come from the
    /// runtime index (see [`index_offsets`]); the other side's are the static residue. Both inner
    /// vectors run over the same free dimension. Shared by `scatter` / `gather`.
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

    /// Reinterprets the physical buffer from `Mapping`-shaped to `Mapping2`-shaped, returning the
    /// result. Backs [`crate::backend::Backend::reshape`]. The wire buffer is shared between the two
    /// layouts, so walk each side's wire plan and copy src canonical cell to dst canonical cell.
    pub(crate) fn reshape<Mapping: M, Mapping2: M>(&self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        let dst_axes = Mapping2::to_value().axes();
        let reads = Mapping::to_value().iter(&self.axes, &Index::new(), true);
        let writes = Mapping2::to_value().iter(&dst_axes, &Index::new(), true);
        let shape: Vec<usize> = dst_axes.iter().map(|term| term.modulo).collect();
        let src_flat = self.flat();
        let mut data = vec![Opt::Uninit; shape.iter().product()];
        for (read, write) in reads.zip(writes) {
            if let (Some(read), Some(write)) = (read, write) {
                data[write] = src_flat[read];
            }
        }
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
        let mut scattered = vec![Opt::Uninit; shape.iter().product()];
        for (offset, &value) in mapping.iter(&axes, &Index::new(), true).zip(&data) {
            if let Some(slot) = offset {
                scattered[slot] = value;
            }
        }
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
        let flat = self.flat();
        mapping
            .iter(&self.axes, &Index::new(), true)
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
        let data = ArrayD::from_elem(IxDyn(&shape), Opt::Uninit);
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
    use furiosa_mapping::*;

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
}
