//! What a backend does, behind [`super::Backend`]: [`Runtime`] opens devices, moves their memory and
//! launches device functions; [`Backend`] adds the tensor storage and ops device functions use.

use super::GatherRead;
use super::indirect::IndexUnit;
use std::fmt::Debug;
use std::future::Future;
use std::ops::RangeBounds;

use furiosa_mapping::*;

use crate::Error;
use crate::cast::{ContractionAccumulator, ContractionCast};
use crate::runtime::{
    DeviceOutput, DeviceSend, HbmBufferSlice, HbmBufferSliceMut, HostBuffer, HostBufferSlice, HostBufferSliceMut, Kind,
    Launched, Placement, Submitted, Topology, Usage,
};
use crate::scalar::{MaterializableScalar, Scalar};
use crate::tensor::memory::{DmAllocError, DmTensor};

/// A backend's device memory: `size()` bytes at the same address on each of `chips()` chips.
pub trait Allocation: Debug + Send + Sync + Sized {
    /// Bytes on each chip.
    fn size(&self) -> usize;

    /// Chips this memory spans, ranks `0..` this many.
    fn chips(&self) -> usize;

    /// Where the bytes start on each chip.
    fn addr(&self) -> usize;

    /// Two owners: every chip's bytes before `offset`, and from it on.
    fn split_at(self, offset: usize) -> Result<(Self, Self), Error>;

    /// Another owner of `bytes` of every chip's range, sharing them without a copy.
    fn alias(&self, bytes: impl RangeBounds<usize>) -> Result<Self, Error>;

    /// Extends this buffer only by an adjacent range of the same allocation on every chip.
    fn join(&mut self, other: &Self) -> bool;

    /// Resolves once the device work submitted before this call that writes these bytes ended.
    fn ready(&self) -> impl Future<Output = Result<(), Error>> + Send + use<Self>;
}

/// Where device functions run and their memory lives: opening a device, allocating and moving its
/// memory, and loading and launching its functions.
pub trait Runtime: Sized + 'static {
    /// An opened device: its memory and, on real chips, the functions loaded on it.
    type Device: Clone + Debug;

    /// What a loaded device function is known by.
    type Kernel: Copy + Debug + Send + Sync;

    /// Device memory: one byte range at the same offset on every chip, which no `Clone` duplicates.
    type Buffer: Allocation;

    /// Opens the first available candidate; an empty list is invalid.
    fn open(placements: Vec<Placement>) -> Result<Self::Device, Error>;

    /// The placement retained by the opened device.
    fn placement(device: &Self::Device) -> &Placement;

    /// The chips and PEs `device` holds.
    fn topology(device: &Self::Device) -> Topology;

    /// Bytes per chip `device` can allocate in all, live or free.
    fn capacity(device: &Self::Device) -> usize;

    /// Free bytes per chip on `device` now.
    fn usage(device: &Self::Device) -> Usage;

    /// A buffer of `bytes` uninitialized bytes on every chip of `device`, of DRAM `kind`:
    /// [`Kind::DramShared`] for memory another chip's device code writes, [`Kind::Dram`] otherwise.
    fn alloc(device: &Self::Device, kind: Kind, bytes: usize) -> Result<Self::Buffer, Error>;

    /// Copies a batch in one runtime write, each host's bytes chip after chip in rank order;
    /// the caller checked every length, and foreign buffers are refused by the backend.
    fn write<'a>(
        device: &Self::Device,
        pairs: Vec<(HostBufferSlice<'a>, HbmBufferSliceMut<'a, Self>)>,
    ) -> impl Future<Output = Result<(), Error>>;

    /// Copies `from` into `host`, chip after chip in rank order; the caller checked that `host`
    /// holds exactly those bytes.
    fn read(
        device: &Self::Device,
        from: HbmBufferSlice<'_, Self>,
        host: HostBufferSliceMut<'_>,
    ) -> impl Future<Output = Result<(), Error>>;

    /// Resolves once every launch and transfer submitted on `device` before this call ended; the
    /// device's failure once it failed.
    fn synchronize(device: &Self::Device) -> impl Future<Output = Result<(), Error>> + Send + use<Self>;

    /// Finds the device function `path`/`key` and loads it on `device`.
    fn load(
        device: &Self::Device,
        path: &'static str,
        key: furiosa_opt_rt::registry::Key<'_>,
    ) -> Result<Self::Kernel, Error>;

    /// Runs `kernel` with `args` on `device`, writing its returns into `output`: the compiled
    /// program where there is a device, `body` (its Rust body) on the host. A device `kernel` was
    /// not loaded on loads it first.
    fn launch<Args: DeviceSend<Self>, Output: DeviceOutput<Self>>(
        device: &mut crate::Device<Self>,
        kernel: Self::Kernel,
        args: Args,
        output: &mut Output,
        body: impl FnOnce(&mut crate::Device<Self>, Args) -> Output,
    ) -> Submitted<Launched<()>>;
}

/// Backend for tensor operations.
///
/// A backend picks the concrete tensor storage type (`Self::Storage<D>`) and the DMA
/// protocol. The storage type is opaque (`Clone + Debug` only); every operation on it, both the
/// lifecycle constructors/serializers and the value-ops (map, reduce, zip_with, scatter, ...),
/// is a method on this trait, delegating to the concrete storage's inherent methods.
///
/// `Cpu`: interprets ops on `BufStorage`'s physical staging buffer (default).
/// `Npu`: owns native staging buffers and does real PCIe DMA; the host never interprets a device
/// function's body.
///
/// Host transfers copy a storage's [`Self::bytes`] with [`Runtime::write`] and [`Runtime::read`],
/// unchanged on both: a host tensor in another layout is relaid out on the host first, by
/// backend-agnostic code.
pub trait Backend: Runtime {
    /// Concrete per-backend tensor type. Opaque:
    /// only `Clone + Debug + PartialEq` (so the memory-tier `#[derive(Debug)]` structs and the
    /// `Tensor<D, Mapping, B>` wrapper's `Debug`/`PartialEq` resolve without a per-backend `where`
    /// clause). `PartialEq` (not `Eq`) because `f32` tensors must compare. All operations on the
    /// storage are methods on this trait.
    type Storage<D: Scalar>: Clone + Debug + PartialEq + From<HostBuffer> + Into<HostBuffer>;

    /// `memory`'s bytes, chip after chip, as `D` values, for a device function that reads device
    /// memory without a DMA context. Only the host backend interprets device functions; a device
    /// backend panics, since its compiled program performs the access on the device.
    fn load_memory<D: Scalar>(memory: &HbmBufferSlice<'_, Self>) -> Self::Storage<D>;

    /// Memory for a tensor a device function makes: `bytes` on each of `chips` chips that no other
    /// owner names, as the compiled program's own memory is.
    fn scratch(chips: usize, bytes: usize) -> Self::Buffer;

    /// Build a storage from a flat `D` buffer in `mapping`-order, delegating to the concrete
    /// storage's inherent constructor. The runtime `mapping` carries the layout, construction is
    /// the type→runtime boundary, so the wrapper converts its type param to a value here.
    fn from_vec<D: Scalar>(mapping: &Mapping, data: impl IntoIterator<Item = D>) -> Self::Storage<D>;

    /// Build a storage from its dense physical device byte image (`mapping`-order, packed on `D::BITS`):
    /// the inverse of [`Self::into_buf`]. The default decodes the packed bytes to a logical `Vec<D>` and
    /// packs via [`Self::from_vec`]; a backend whose storage already IS the packed image (Cpu /
    /// Npu's [`crate::storage::BufStorage`]) overrides this to store the bytes directly, no decode / re-pack. Pre-packed
    /// data (fp4 / f4e2m1 weights) enters through here. Backs [`crate::tensor::Tensor::from_buf`].
    fn from_buf<D: MaterializableScalar>(mapping: &Mapping, buf: Vec<u8>) -> Self::Storage<D> {
        Self::from_vec(mapping, (0..mapping.size()).map(|i| D::load(&buf, i)))
    }

    /// A zeroed storage of this layout, delegating to the concrete storage's inherent
    /// constructor.
    fn zeroed<D: Scalar>(mapping: &Mapping) -> Self::Storage<D>;

    /// A runtime-owned DM tensor on `group` (see
    /// [`crate::tensor::memory::DmTensor::alloc`]). `Npu` takes a resident SRAM allocation the
    /// tensor owns; the host-side backends hold bytes.
    fn alloc_dm<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        group: &Self::Device,
    ) -> Result<DmTensor<D, Chip, Cluster, Slice, Element, Self>, DmAllocError>;

    /// Copies initialized prefix cells from a table laid out as `mapping` through its index.
    fn gather_table<D: MaterializableScalar, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        mapping: &Mapping,
        indexed: &Mapping,
        chip: &Mapping,
        read: GatherRead<'_, Self::Storage<i32>>,
    );

    /// Updates indexed payloads of a table laid out as `mapping`, preserving its other cells.
    fn scatter_table<D: Scalar, Src: M, Key: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        mapping: &Mapping,
        indexed: &Mapping,
        chip: &Mapping,
        unit: IndexUnit,
    );

    /// Serialize the storage to a flat `D` buffer in `mapping`-order (the physical / wire layout),
    /// consuming the storage, delegating to the concrete storage's inherent serializer.
    fn into_vec<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &Mapping) -> Vec<D>;

    /// Serialize the storage to its dense physical device byte image (`mapping`-order, packed on
    /// `D::BITS`): the exact bytes that sit in HBM / DM, half the logical length for a 4-bit `D`. The
    /// default decodes to a logical `Vec<D>` and re-packs via [`Scalar::to_buf`]; a backend whose
    /// storage already IS the packed image (Cpu / Npu's [`crate::storage::BufStorage`]) overrides this to move the
    /// internal bytes out directly. Backs [`crate::tensor::memory::HbmTensor::to_buf`].
    fn into_buf<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &Mapping) -> Vec<u8> {
        D::to_buf(&Self::into_vec(storage, mapping))
    }

    /// Element-wise map to a new scalar, preserving layout. The closure is bare-`D`; each backend
    /// applies its own per-storage strategy; Cpu/Npu walk the `BufStorage` buffer over bare
    /// `D`. Mapping-free: elementwise ops preserve the source
    /// layout, so no mapping type parameter is needed.
    /// The closure must be `Fn + Sync` (not `FnMut`): an elementwise map has no cross-cell state, and
    /// [`Cpu`] folds the cells across the rayon pool.
    fn map<D: Scalar, D2: Scalar>(src: &Self::Storage<D>, f: impl Fn(D) -> D2 + Sync) -> Self::Storage<D2>;

    /// Element-wise zip of two same-layout tensors to a new scalar. See [`Self::map`]; the result is
    /// defined only where both inputs are.
    fn zip_with<D: MaterializableScalar, D2: MaterializableScalar, D3: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> Self::Storage<D3>;

    /// Element-wise zip of three same-layout tensors to a new scalar, the ternary peer of
    /// [`Self::map`] (1-ary) and [`Self::zip_with`] (2-ary). The result is defined only where all
    /// three inputs are. The VE engine uses it for branch-conditional combine: it passes the
    /// execution-id `tag` as the third operand and resolves the slot's `TagGuard`/select inside the closure,
    /// so the backend stays free of VE concepts.
    fn zip3_with<D: MaterializableScalar, D2: MaterializableScalar, D3: MaterializableScalar, D4: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        c: &Self::Storage<D3>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> Self::Storage<D4>;

    /// Writes a transposed/broadcast view of `src` into `dst`. `src_offset` / `dst_offset` allow
    /// operating on partial views; `src_map` / `dst_map` are each storage's base (live-axis) mapping,
    /// where a tiled axis the `Src` / `Dst` *type* carries as padding stays a live `Symbol`, needed
    /// to resolve a partial-view offset's physical wire base. Each backend applies its own per-storage
    /// strategy: Cpu/Npu resolve the offset bases against the maps and drive a
    /// sequencer over the `BufStorage` buffer.
    #[allow(clippy::too_many_arguments)]
    fn transpose<D: Scalar, Src: M, Dst: M>(
        dst: &mut Self::Storage<D>,
        src: &Self::Storage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &Mapping,
        dst_map: &Mapping,
        allow_broadcast: bool,
        preserve_copies: bool,
    );

    /// Reduces the factors of `src`'s mapping absent in `Dst`, returning the result. Axes that `Dst`
    /// adds beyond the reduced source are broadcast (the reduced value repeats across them).
    /// `allow_broadcast` gates that, exactly like [`Self::transpose`]'s flag: `false` is a
    /// plain reduce that asserts `Dst` is a pure factor and *rejects* an unintended extra axis;
    /// `true` permits the broadcast axes. The reduce fn works over bare scalars; the backend lifts
    /// it to its cell representation.
    ///
    /// `reduce_fn` should be associative: the parallel backends may regroup the fold across threads. An
    /// associative fn gives a deterministic, serial-identical result; a non-associative one (saturating
    /// integer add, float add/mul) can differ from a serial left-fold and vary run-to-run on
    /// [`Cpu`], which regroups the fold across the rayon pool.
    fn reduce<D: MaterializableScalar, Src: M, Dst: M>(
        src: &Self::Storage<D>,
        reduce_fn: impl Fn(D, D) -> D + Sync,
        identity: D,
        allow_broadcast: bool,
    ) -> Self::Storage<D>;

    /// Contracts `lhs` and `rhs` into an `out`-shaped result: a generalized matmul
    /// `out[..] = sum over the axes of pre_reduce absent from out of (lhs * rhs)`. `pre_reduce` is the
    /// full (un-reduced) index space; `lhs_map` / `rhs_map` / `out` are its sub-mappings (the contracted
    /// axes are those in `pre_reduce` but not `out`).
    ///
    /// A required primitive, not a default: every backend fuses this into a single accumulating pass
    /// over `out`-many cells (`O(out)` memory), never materializing the `pre_reduce`-shaped outer
    /// product (`O(pre_reduce)`, which for a wide-output GEMM is the hundreds-of-MB dense buffer this
    /// primitive exists to avoid). The mappings arrive as runtime values because the engine-level
    /// contraction pipeline ([`crate::engine::contraction`]) only has them as values at the point it
    /// defers the fold; `lhs`/`rhs` keep their own compact (un-broadcast) layout, read through
    /// `lhs_map`/`rhs_map`.
    ///
    /// `pre_reduce` is not simply "the union of `lhs_map`'s and `rhs_map`'s axes": the engine-level
    /// caller can reconcile the two operands' contracted portions through DIFFERENT symbols (e.g. one
    /// operand's `Time`/`Packet` against the other's `TrfElement`, per the hardware layout-expansion
    /// contract `contract_outer` validates), which a naive per-symbol union of the two maps cannot
    /// recover. So it is a required, independently-computed argument, not derived here.
    ///
    /// Like [`Self::reduce`], [`Cpu`]'s per-output-cell sum regroups across threads, so its
    /// `f32` result can differ from serial and vary run-to-run.
    ///
    /// Operands arrive at the engine's operand width and accumulate in [`ContractionCast::Output`].
    /// [`Self::contraction_prewidened`] is the same fold over operands already at that width.
    fn contraction<D: ContractionCast + MaterializableScalar>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &Mapping,
        rhs_map: &Mapping,
        pre_reduce: &Mapping,
        out: &Mapping,
    ) -> Self::Storage<D>;

    /// [`Self::contraction`] over operands already at [`ContractionAccumulator`] width, so its widen and
    /// narrow are the identity. This is the pipeline's own fold, at `contract_lane`.
    fn contraction_prewidened<D: ContractionAccumulator>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &Mapping,
        rhs_map: &Mapping,
        pre_reduce: &Mapping,
        out: &Mapping,
    ) -> Self::Storage<D>;

    /// Scatters `src` into `dst` at positions read from the same-backend `i32` index tensor.
    fn scatter<D: Scalar, Src: M, Domain: M, IndexedAxis: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        chip: &Mapping,
        unit: IndexUnit,
    );

    /// Gathers from `src` along `IndexedAxis`, validating every index entry.
    /// `read.prefix` limits output entries per chip; `None` gathers the whole list.
    fn gather<D: MaterializableScalar, Src: M, IndexedAxis: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        chip: &Mapping,
        read: GatherRead<'_, Self::Storage<i32>>,
    );

    /// Reinterprets `src`'s buffer under a new mapping `Dst`, returning the result. A relabeling, not
    /// a data move: `BufStorage` keeps the same physical buffer, pairing wire position `i` under
    /// `Src` with wire position `i` under `Dst`.
    ///
    /// Memory-safe and length-checked (asserts `Src::SIZE == Dst::SIZE`). Whether the relabel is
    /// *meaningful* is the caller's logical precondition, discharged by the `unsafe` public `reshape`
    /// wrappers: `Src` and `Dst` must enumerate the elements in the SAME physical (wire) order, i.e.
    /// `Src::indexes()` and `Dst::indexes()` line the live elements up identically. Regrouping axes
    /// (merge/split) preserves wire order and is valid; a permutation (e.g. `m![A, B]` → `m![B, A]`)
    /// does NOT and must use `transpose`. This method itself only guarantees memory safety.
    fn reshape<D: Scalar, Src: M, Dst: M>(src: &Self::Storage<D>) -> Self::Storage<D>;

    /// Relabels `storage` from `src_map` to `dst_map` (`Tensor::transmute`). Unlike `reshape`, the two
    /// maps may differ in size: the vector engine relabels a padded read down to its compact slot
    /// (`…tile(k).read().transmute()`), so `dst_map.size() <= src_map.size()` is intended.
    ///
    /// `Cpu` / `Npu` COMPACT the
    /// physical buffer: they keep only the live (non-padding) cells of the `src_map` layout, in
    /// `dst_map` order, producing a `dst_map.size()`-length buffer (a no-op when no padding is dropped).
    fn transmute<D: MaterializableScalar, Src: M, Dst: M>(
        storage: Self::Storage<D>,
        src_map: &Mapping,
        dst_map: &Mapping,
    ) -> Self::Storage<D>;

    /// `storage`'s bytes in place, for a transfer to copy from: the DMA reads page-locked memory
    /// directly.
    fn bytes<D: Scalar>(storage: &Self::Storage<D>) -> HostBufferSlice<'_>;

    /// `storage`'s bytes in place, for a transfer to copy into: the DMA writes page-locked memory
    /// directly.
    fn bytes_mut<D: Scalar>(storage: &mut Self::Storage<D>) -> HostBufferSliceMut<'_>;

    /// Host storage moved to page-locked memory, so transfers use it directly; a backend without a
    /// device has nothing to pin and returns the storage unchanged.
    fn pin<D: MaterializableScalar>(storage: Self::Storage<D>) -> Result<Self::Storage<D>, Error>;
}
