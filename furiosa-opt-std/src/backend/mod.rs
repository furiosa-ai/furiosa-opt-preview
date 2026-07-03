//! Backend abstraction: the storage-op seam for tensor operations.
//!
//! A [`Backend`] picks the concrete tensor storage type and implements every tensor operation
//! against it. The concrete impls are [`Simulation`] (host-side mapping interpreter),
//! [`Emulation`] (host-side buffer emulator), [`Npu`] (real device), and [`Typecheck`]
//! (shape/mapping validation only). The cfg-selected [`crate::runtime::CurrentBackend`] alias picks which one the
//! crate compiles against. Device-function execution (`launch`, `DeviceFn`, `DeviceSend`) lives in
//! the sibling [`crate::runtime`] module.

pub(crate) mod op_prep;

mod emulation;
/// NPU backend.
pub mod npu;
mod simulation;
mod typecheck;

use std::fmt::Debug;

use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::cast::ContractionCast;
use crate::scalar::Scalar;
use crate::tensor::memory::{HbmTensor, HostTensor};

pub use emulation::Emulation;
pub use npu::Npu;
pub use simulation::Simulation;
pub use typecheck::Typecheck;

/// Backend for tensor operations.
///
/// A backend picks the concrete tensor storage type (`Self::Storage<D>`) and the DMA
/// protocol. The storage type is opaque (`Clone + Debug` only); every operation on it, both the
/// lifecycle constructors/serializers and the value-ops (map, reduce, zip_with, scatter, ...),
/// is a method on this trait, delegating to the concrete storage's inherent methods.
///
/// `Simulation`: interprets ops via mapping expressions on `MathStorage` (default).
/// `Emulation`: interprets ops on `BufStorage`'s physical staging buffer.
/// `Npu`: owns native staging buffers and does real PCIe DMA; the host never interprets ops.
/// `Typecheck`: shape/mapping validation only, no host values (`--cfg backend="typecheck"`).
///
/// Host backends (Simulation, Emulation) implement the DMA as a plain host-side `transpose`
/// passthrough; `Npu` provides real device transfers. DMA stays a required method (no default) so
/// each backend states its choice explicitly.
pub trait Backend: Sized + 'static {
    /// Concrete per-backend tensor type (`MathStorage` / `BufStorage` / `PhantomStorage`). Opaque:
    /// only `Clone + Debug + PartialEq` (so the memory-tier `#[derive(Debug)]` structs and the
    /// `Tensor<D, Mapping, B>` wrapper's `Debug`/`PartialEq` resolve without a per-backend `where`
    /// clause). `PartialEq` (not `Eq`) because `f32` tensors must compare. All operations on the
    /// storage are methods on this trait.
    type Storage<D: Scalar>: Clone + Debug + PartialEq;

    /// Build a storage from a flat `D` buffer in `mapping`-order, delegating to the concrete
    /// storage's inherent constructor. The runtime `mapping` carries the layout, construction is
    /// the type→runtime boundary, so the wrapper converts its type param to a value here.
    fn from_buf<D: Scalar>(mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self::Storage<D>;

    /// An uninitialized storage of this layout, delegating to the concrete storage's inherent
    /// constructor.
    fn uninit<D: Scalar>(mapping: &MappingValue) -> Self::Storage<D>;

    /// Serialize the storage to a flat `D` buffer in `mapping`-order (the physical / wire layout),
    /// consuming the storage, delegating to the concrete storage's inherent serializer.
    fn into_buf<D: Scalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<D>;

    /// Element-wise map to a new scalar, preserving layout. The closure is bare-`D`; each backend
    /// applies its own per-storage strategy, Simulation walks `MathStorage`'s dense array and
    /// propagates `Uninit`, Emulation/Npu walk the `BufStorage` buffer over bare `D`, Typecheck
    /// no-ops on the phantom. Mapping-free: elementwise ops preserve the source layout, so no mapping
    /// type parameter is needed.
    /// The closure must be `Fn + Sync` (not `FnMut`): an elementwise map has no cross-cell state, and
    /// [`Simulation`] folds the cells across the rayon pool.
    fn map<D: Scalar, D2: Scalar>(src: &Self::Storage<D>, f: impl Fn(D) -> D2 + Sync) -> Self::Storage<D2>;

    /// Element-wise zip of two same-layout tensors to a new scalar. See [`Self::map`]; the result is
    /// defined only where both inputs are.
    fn zip_with<D: Scalar, D2: Scalar, D3: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> Self::Storage<D3>;

    /// Element-wise zip of three same-layout tensors to a new scalar, the ternary peer of
    /// [`Self::map`] (1-ary) and [`Self::zip_with`] (2-ary). The result is defined only where all
    /// three inputs are. The VE engine uses it for branch-conditional combine: it passes the
    /// execution-id `tag` as the third operand and resolves `TagFilter`/select inside the closure,
    /// so the backend stays free of VE concepts.
    fn zip3_with<D: Scalar, D2: Scalar, D3: Scalar, D4: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        c: &Self::Storage<D3>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> Self::Storage<D4>;

    /// Writes a transposed/broadcast view of `src` into `dst`. `src_offset` / `dst_offset` allow
    /// operating on partial views; `src_map` / `dst_map` are each storage's base (live-axis) mapping,
    /// where a tiled axis the `Src` / `Dst` *type* carries as padding stays a live `Symbol`, needed
    /// to resolve a partial-view offset's physical wire base. Each backend applies its own per-storage
    /// strategy: Simulation addresses `MathStorage` by coordinate (the maps are unused, its storage
    /// already carries them), Emulation/Npu resolve the offset bases against the maps and drive a
    /// sequencer over the `BufStorage` buffer, Typecheck runs only the structural check.
    #[allow(clippy::too_many_arguments)]
    fn transpose<D: Scalar, Src: M, Dst: M>(
        dst: &mut Self::Storage<D>,
        src: &Self::Storage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &MappingValue,
        dst_map: &MappingValue,
        allow_broadcast: bool,
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
    /// [`Emulation`] (the [`Simulation`] per-output-cell fold stays serial-identical).
    fn reduce<D: Scalar, Src: M, Dst: M>(
        src: &Self::Storage<D>,
        reduce_fn: impl Fn(D, D) -> D + Sync,
        identity: D,
        allow_broadcast: bool,
    ) -> Self::Storage<D>;

    /// Contracts `lhs` and `rhs` into an `Out`-shaped result: a generalized matmul
    /// `out[..] = sum over the axes of Union absent from Out of (lhs * rhs)`. `Union` is the full
    /// index space; `Lhs` / `Rhs` / `Out` are its sub-mappings (the contracted axes are those in
    /// `Union` but not `Out`).
    ///
    /// The default materializes `Union`: it broadcasts both operands into the `Union` shape,
    /// multiplies, then reduces, costing O(`Union`) memory. The host backends ([`Simulation`],
    /// [`Emulation`]) override this with a fused single-pass O(`Out`)-memory version, which also lets
    /// contractions far larger than memory's `Union` bound run.
    ///
    /// Like [`Self::reduce`], the per-output-cell sum is exact and deterministic on [`Simulation`]
    /// (one worker folds each cell), but [`Emulation`] regroups it across threads, so its `f32` result
    /// can differ from serial and vary run-to-run.
    ///
    /// `D: ContractionCast` is required by the host-backend overrides, whose folds widen to
    /// [`ContractionCast::Output`]; the bound must cover every overriding impl even though this default
    /// folds in narrow `D` and does not use it.
    fn contraction<D: Scalar + ContractionCast, Union: M, Lhs: M, Rhs: M, Out: M>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
    ) -> Self::Storage<D> {
        let union = Union::to_value();
        let mut lhs_u = Self::uninit::<D>(&union);
        Self::transpose::<D, Lhs, Union>(
            &mut lhs_u,
            lhs,
            &Index::new(),
            &Index::new(),
            &Lhs::to_value(),
            &union,
            true,
        );
        let mut rhs_u = Self::uninit::<D>(&union);
        Self::transpose::<D, Rhs, Union>(
            &mut rhs_u,
            rhs,
            &Index::new(),
            &Index::new(),
            &Rhs::to_value(),
            &union,
            true,
        );
        let prod = Self::zip_with::<D, D, D>(&lhs_u, &rhs_u, |a, b| a * b);
        Self::reduce::<D, Union, Out>(&prod, |a, b| a + b, num_traits::Zero::zero(), false)
    }

    /// Scatters `src` into `dst` at positions read from the same-backend `i32` index tensor.
    fn scatter<D: Scalar, Src: M, Key: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    );

    /// Gathers from `src` (table) into `dst` at positions read from the index tensor.
    fn gather<D: Scalar, Src: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    );

    /// Reinterprets `src`'s buffer under a new mapping `Dst`, returning the result. A relabeling, not
    /// a data move: `BufStorage` keeps the same physical buffer; `MathStorage` rebuilds it pairing
    /// wire position `i` under `Src` with wire position `i` under `Dst`.
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
    /// `Simulation` / `Typecheck` return `storage` unchanged, their canonical storage is already
    /// padding-free, so a smaller `dst_map` needs no data move. `Emulation` / `Npu` COMPACT the
    /// physical buffer: they keep only the live (non-padding) cells of the `src_map` layout, in
    /// `dst_map` order, producing a `dst_map.size()`-length buffer (a no-op when no padding is dropped).
    fn transmute<D: Scalar, Src: M, Dst: M>(
        storage: Self::Storage<D>,
        src_map: &MappingValue,
        dst_map: &MappingValue,
    ) -> Self::Storage<D>;

    /// Transfer host tensor to HBM. Host-only: invoked from test/setup code over PCIe DMA, never
    /// from device kernel MIR, so no `#[primitive(...)]` annotation is needed.
    fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
    ) -> impl std::future::Future<Output = HbmTensor<D, Chip, Element2, Self>>;

    /// Transfer HBM tensor to host. Host-only; see the `to_hbm` note above.
    fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> impl std::future::Future<Output = HostTensor<D, Element2, Self>>;

    /// Bind the process to a logical [`Device`](crate::context::Device) before any host I/O. Only the
    /// NPU backend acts on this; backends without a physical device default to a no-op.
    fn bind_device(_device: crate::context::Device) {}
}
