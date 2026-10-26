//! Tensors placed on memory.

mod dma_layout;
mod gather;
mod hbm_scalar;
mod redistribute;
mod scatter;

pub use gather::*;
pub use hbm_scalar::HbmScalar;
pub use redistribute::*;
pub use scatter::*;

use dma_layout::{assert_dm_dma_layout, assert_dma_layout};
use rand::Rng;
use rand::distr::StandardUniform;
use std::future::Future;
use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_lower::{DM_WRITE_ALIGN_BYTES, PadInput, TileInput, config_pad, config_tile};
use furiosa_opt_macro::primitive;

use crate::Error;
use crate::backend::{Allocation, Backend, Cpu};
use crate::constraints;
use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::{
    CurrentBackend, HbmBufferSlice, HbmBufferSliceMut, HostBuffer, Issued, Prepared, Region, Submitted, Work,
};
use crate::scalar::*;
use crate::tensor::view::Tiled;
use crate::tensor::*;

/// Address.
///
/// TODO: check that every address is 64-bit.
pub type Address = u64;

/// Tensor stored in host memory.
///
/// `D: MaterializableScalar`, not just `Scalar`: a `HostTensor` is exactly the "host... carrier" the
/// bound's own doc names, so a non-materializable staging type (`i5`/`i9`) can never even be
/// CONSTRUCTED as one -- rejected at the type's own definition, not left to be caught only if some
/// particular method (`into_vec`, etc.) happens to touch the unsound whole-buffer path. `i5`/`i9`'s
/// entire legitimate lifetime is between `fetch_zero_point_sub` and `contract_outer`, both
/// engine-internal (`TuTensor`/bare `Tensor`), never through this public host-facing wrapper.
#[primitive(HostTensor)]
#[derive(Debug, Clone)]
pub struct HostTensor<D: MaterializableScalar, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Element, B>,
}

impl<D: MaterializableScalar, Element: M, B: Backend> From<Tensor<D, Element, B>> for HostTensor<D, Element, B> {
    fn from(inner: Tensor<D, Element, B>) -> Self {
        Self { inner }
    }
}

/// A host buffer's bytes read as `D` values laid out as `Element`, once its length is the
/// tensor's; [`HostTensor::into_buffer`] gives the buffer back.
impl<D: MaterializableScalar, Element: M, B: Backend> TryFrom<HostBuffer> for HostTensor<D, Element, B> {
    type Error = Error;

    fn try_from(buffer: HostBuffer) -> Result<Self, Error> {
        if buffer.len() != D::size_in_bytes_from_length(Element::SIZE) {
            return Err(Error::Invalid("the host buffer's length is not the tensor's"));
        }
        Ok(Tensor::from_inner(buffer.into()).into())
    }
}

impl<D: MaterializableScalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = Element;

    pub(crate) fn inner(&self) -> &Tensor<D, Element, B> {
        &self.inner
    }

    pub(crate) fn storage(&self) -> &B::Storage<D> {
        &self.inner.inner
    }

    pub(crate) fn storage_mut(&mut self) -> &mut B::Storage<D> {
        &mut self.inner.inner
    }

    /// The same tensor in page-locked host memory, so a transfer reads or writes it directly and
    /// asynchronously. Already pinned memory stays.
    ///
    /// # Errors
    /// [`Error::Device`] when the system cannot page-lock the memory.
    pub fn pinned(self) -> Result<Self, Error> {
        Ok(Tensor::from_inner(B::pin(self.inner.inner)?).into())
    }

    /// Creates a tensor from an initialized buffer. Panics if the buffer length does not match the
    /// mapping size.
    pub fn from_vec(data: impl IntoIterator<Item = D>) -> Self {
        Tensor::from_vec(data).into()
    }

    /// Creates a tensor from a pre-packed device byte image ([`crate::scalar::Scalar::to_buf`]'s inverse), stored as-is.
    /// Contrast [`Self::from_vec`], which packs logical values; pre-packed fp4 weights come through here
    /// to avoid a decode + re-pack round-trip. Panics on a byte-length mismatch.
    pub fn from_buf(buf: Vec<u8>) -> Self {
        Tensor::from_buf(buf).into()
    }

    /// Prepares a transfer of this tensor to HBM: `.await` stages it into a fresh device
    /// allocation, `.output(&mut hbm).await` writes into an existing one.
    pub fn to_hbm<'t, Chip: M, Element2: M>(
        &'t self,
        dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    ) -> Prepared<ToHbm<'t, D, Element, Chip, Element2, B>> {
        Prepared::new(ToHbm {
            host: self,
            dma,
            _marker: PhantomData,
        })
    }

    /// The host buffer this tensor's bytes live in, as the device holds them.
    pub fn into_buffer(self) -> HostBuffer {
        self.inner.inner.into()
    }

    /// Consumes self and returns the inner tensor.
    pub fn into_inner(self) -> Tensor<D, Self::Mapping, B> {
        self.inner
    }

    /// Returns the tensor data as a flat `Vec<D>`, consuming the tensor.
    pub fn into_vec(self) -> Vec<D> {
        self.inner.into_vec()
    }
}

/// Host-side `HostTensor` constructors. Bound to `Backend`; the value-iterating methods (`zero`,
/// `rand`) and `from_vec` / `from_safetensors` all bottom out in `Tensor::from_vec`, which
/// `BufStorage` implements as a real `Vec<D>` fill, so those work on Npu / Cpu host-side
/// staging too.
impl<D: MaterializableScalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Creates a tensor filled with zeros.
    pub fn zero() -> Self
    where
        D: num_traits::Zero,
    {
        Tensor::splat(D::zero()).into()
    }

    /// Creates a tensor filled with random values.
    #[primitive(HostTensor::rand)]
    pub fn rand(rng: &mut impl Rng) -> Self
    where
        StandardUniform: rand::distr::Distribution<D>,
    {
        Tensor::rand(rng).into()
    }

    /// Creates a tensor from a `safetensors` tensor view.
    ///
    /// The view's per-axis shape must match `Element`'s pair-flattened size list (e.g.
    /// `m![H, X]` expects safetensors shape `[H.size, X.size]`) and its bytes are decoded as
    /// little-endian `D` values — LE is mandated by the safetensors format spec, not our
    /// choice. Returns [`safetensors::SafeTensorError::TensorInvalidInfo`] on any mismatch.
    pub fn from_safetensors(view: &safetensors::tensor::TensorView<'_>) -> Result<Self, safetensors::SafeTensorError>
    where
        D: ScalarBytes,
    {
        fn flat_shape(mapping: &Mapping, out: &mut Vec<usize>) {
            match mapping {
                Mapping::Pair { left, right } => {
                    flat_shape(left, out);
                    flat_shape(right, out);
                }
                _ => out.push(mapping.size()),
            }
        }
        let mut expected_shape = Vec::new();
        flat_shape(&Element::to_value(), &mut expected_shape);
        if view.shape() != expected_shape.as_slice() {
            return Err(safetensors::SafeTensorError::TensorInvalidInfo);
        }
        let stride = D::BITS / 8;
        if view.data().len() != Element::SIZE * stride {
            return Err(safetensors::SafeTensorError::TensorInvalidInfo);
        }
        // The view's LE bytes ARE the packed device image for a byte-multiple `D`, so store them directly
        // through `from_buf` rather than decoding each element and re-packing.
        Ok(Tensor::from_buf(view.data().to_vec()).into())
    }
}

/// Tensor stored in HBM memory: device memory with a type.
///
/// `TryFrom<crate::HbmBuffer>` takes a buffer whose per-chip size, chip count and alignment fit the
/// type; [`Self::into_buffer`] gives it back without copying.
#[primitive(HbmTensor)]
pub struct HbmTensor<D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend> {
    /// The memory this tensor names and keeps live: one chip's share of the bytes on every chip.
    buffer: B::Buffer,
    shape: PhantomData<(D, Chip, Element)>,
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> std::fmt::Debug for HbmTensor<D, Chip, Element, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HbmTensor")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> TryFrom<crate::HbmBuffer<B>> for HbmTensor<D, Chip, Element, B> {
    type Error = Error;

    fn try_from(buffer: crate::HbmBuffer<B>) -> Result<Self, Error> {
        if tensor_bytes::<D, Chip, Element>(&buffer.0, 0)? != buffer.0.size() {
            return Err(Error::Invalid("the buffer is larger than the tensor"));
        }
        Ok(Self::from_buffer(buffer.0))
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Element }];

    fn from_buffer(buffer: B::Buffer) -> Self {
        Self {
            buffer,
            shape: PhantomData,
        }
    }

    /// A fresh HBM tensor in a device function, in memory the compiled program owns.
    // `new()` builds an uninitialized handle, so a `Default` impl (which would look zero-cost and
    // safe) is deliberately not provided, matching `DmTensor::new`.
    #[allow(clippy::new_without_default)]
    #[primitive(HbmTensor::new)]
    pub fn new() -> Self {
        let bytes = D::size_in_bytes_from_length(Element::SIZE);
        Self::from_buffer(B::scratch(Chip::SIZE, bytes))
    }

    /// `value` in memory a device function makes, as the device function writes it.
    pub(crate) fn from_value<const DMA: Dma>(dma: &DmaContext<DMA, B>, value: &Tensor<D, Self::Mapping, B>) -> Self {
        let mut tensor = Self::new();
        tensor.store(dma, value);
        tensor
    }

    /// The memory this tensor names.
    pub(crate) fn memory(&self) -> HbmBufferSlice<'_, B> {
        HbmBufferSlice::whole(&self.buffer)
    }

    pub(crate) fn memory_mut(&mut self) -> HbmBufferSliceMut<'_, B> {
        HbmBufferSliceMut::whole(&mut self.buffer)
    }

    /// The tensor's values, as a device function reads them.
    pub(crate) fn read<const DMA: Dma>(&self, dma: &DmaContext<DMA, B>) -> Tensor<D, Self::Mapping, B> {
        Tensor::from_inner(dma.load(self.memory()))
    }

    /// Overwrites the tensor with `value`, as a device function writes it.
    pub(crate) fn store<const DMA: Dma>(&mut self, dma: &DmaContext<DMA, B>, value: &Tensor<D, Self::Mapping, B>) {
        dma.store(self.memory_mut(), &value.inner);
    }

    /// A tensor in a new, uninitialized allocation on `device`, which each chip's device code
    /// reaches only on its own chip, as [`HbmBuffer::alloc`](crate::HbmBuffer::alloc) places it.
    pub fn alloc(device: &crate::Device<B>) -> Result<Self, Error> {
        Self::alloc_in(device.pdma.device(), crate::runtime::Kind::Dram)
    }

    pub(crate) fn alloc_in(device: &B::Device, kind: crate::runtime::Kind) -> Result<Self, Error> {
        let bytes = D::size_in_bytes_from_length(Element::SIZE);
        crate::HbmBuffer(B::alloc(device, kind, bytes)?).try_into()
    }

    /// Resolves once the launches and transfers submitted before this call that write this tensor
    /// ended, without copying it.
    pub fn ready(&self) -> impl Future<Output = Result<(), Error>> + Send + use<D, Chip, Element, B> {
        self.buffer.ready()
    }

    /// The buffer this tensor's bytes live in.
    pub fn into_buffer(self) -> crate::HbmBuffer<B> {
        crate::HbmBuffer(self.buffer)
    }

    /// Size of the packed device image in bytes.
    ///
    /// The wire width, not the staging width: the types staged wider (`i5` / `i9`) are not
    /// `MaterializableScalar`, so no host-facing buffer ever holds one.
    pub fn size() -> usize {
        D::size_in_bytes_from_length(Pair::<Chip, Element>::SIZE)
    }

    /// Prepares a transfer of this tensor to the host, which waits for the launches and transfers
    /// submitted before it that write the tensor: `.await` reads it into fresh host memory,
    /// `.output(&mut host).await` reads into an existing tensor, directly when that tensor is
    /// [`HostTensor::pinned`].
    pub fn to_host<'t, Element2: M>(
        &'t self,
        dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    ) -> Prepared<ToHost<'t, D, Chip, Element, Element2, B>>
    where
        D: MaterializableScalar,
    {
        Prepared::new(ToHost {
            hbm: self,
            dma,
            _marker: PhantomData,
        })
    }
}

impl<D: MaterializableScalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Creates an immutable view of the tensor.
    #[primitive(HbmTensor::view)]
    pub fn view<'l>(&'l self) -> HbmTensorView<'l, D, Chip, Element, B> {
        HbmTensorView {
            inner: View::whole(self.memory()),
            chips: PhantomData,
        }
    }

    /// Creates a mutable view of the tensor.
    #[primitive(HbmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> HbmTensorViewMut<'l, D, Chip, Element, B> {
        HbmTensorViewMut {
            inner: View::whole(self.memory_mut()),
            chips: PhantomData,
        }
    }

    /// Redistributes HBM chip slots by DMA: `shuffle_pattern[target] = source`.
    #[primitive(HbmTensor::hbm_chip_shuffle)]
    pub fn hbm_chip_shuffle<const CHIP_DIM: usize, const DMA: Dma>(
        &self,
        dma: &mut DmaContext<{ DMA }, B>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> HbmTensor<D, Chip, Element, B> {
        self.view().hbm_chip_shuffle(dma, shuffle_pattern)
    }

    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(HbmTensor::to_hbm)]
    pub fn to_hbm<const DMA: Dma, Element2: M>(
        &self,
        dma: &mut DmaContext<{ DMA }, B>,
    ) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::from_value(dma, &self.read(dma).transpose(true))
    }
}

// ANCHOR: dma_impl
impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Converts to data memory tensor.
    #[primitive(HbmTensor::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        &self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        DmTensor::from_parts(self.read(dma).transpose(true))
    }

    /// Reshapes the tensor to a different mapping at the same HBM address, consuming `self`.
    /// The HBM analogue of [`DmTensor::reshape`]; both delegate to [`Tensor::reshape`].
    ///
    /// # Safety
    ///
    /// The per-level sizes (`Chip::SIZE == Chip2::SIZE`, `Element`) are asserted at compile time below
    /// (see [`constraints::assert_hbm_reshape_dimension_preserved`]); the genuine precondition is
    /// [`Tensor::reshape`]'s: the old and new mappings must lay the elements out in the SAME physical
    /// (wire) order, so the relabel moves no data. Axis regrouping (merge/split) preserves wire order
    /// and is valid; a permutation is not (use a transpose). Equal sizes do not guarantee this.
    /// Consuming `self` is the safety contract made explicit: the old-shaped handle cannot survive to
    /// alias the same HBM bytes under a conflicting mapping.
    #[primitive(HbmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Element2: M>(self) -> HbmTensor<D, Chip2, Element2, B> {
        constraints::assert_hbm_reshape_dimension_preserved::<Chip, Chip2, Element, Element2>();
        HbmTensor::from_buffer(self.buffer)
    }
}
// ANCHOR_END: dma_impl

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Shuffles data across clusters on HBM (HBM ↔ HBM DMA). Not yet implemented: an `HbmTensor`
    /// has no cluster dimension (only Chip and Element); clusters are assigned later, at `to_dm`.
    pub fn hbm_cluster_shuffle<const DMA: Dma>(
        &self,
        _dma: &mut DmaContext<{ DMA }, B>,
        _shuffle_pattern: &[usize],
    ) -> Self {
        todo!(
            "hbm_cluster_shuffle is Under Construction. HbmTensor has no Cluster axis \
             (only Chip + Element); Cluster distribution is decided at .to_dm() time. \
             No current callers. Either the Element axis is meant to encode a Cluster \
             sub-axis (API needs to take that axis explicitly) or the operation belongs \
             on DmTensorView::cluster_swap. Pending design review; see the doc \
             comment on hbm_cluster_shuffle."
        )
    }
}

/// View of an HBM tensor, over every chip of it ([`AllChips`]) or over the chips a
/// [`chip_tile`](Self::chip_tile) selects ([`SomeChips`]).
#[primitive(HbmTensorView)]
#[derive(Debug, Clone)]
pub struct HbmTensorView<'l, D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend, Chips = AllChips> {
    /// The view's cells, over the base tensor's device memory.
    inner: View<'l, D, Pair<Chip, Element>, B, HbmBufferSlice<'l, B>>,
    chips: PhantomData<Chips>,
}

/// Marks an HBM view over every chip of its tensor, which a launch binds.
#[derive(Debug, Clone)]
pub struct AllChips;

/// Marks an HBM view that may cover only some chips of its tensor, as a
/// [`chip_tile`](HbmTensorView::chip_tile) makes. Device DMA reads and writes it, but a launch binds
/// every chip of an argument, so it refuses one:
///
/// ```compile_fail,E0277
/// use furiosa_opt_std::prelude::*;
/// axes![A = 16, C = 4];
/// fn bind(_: impl DeviceSend) {}
/// let tensor = HbmTensor::<i32, m![C], m![A]>::new();
/// bind(tensor.view().chip_tile::<m![C], 1, m![1 # 4]>(1));
/// ```
#[derive(Debug, Clone)]
pub struct SomeChips;

/// The per-chip byte range a view covers: from the window starting at element `window_base` to the
/// end of a base tensor of `base_len` elements spread over `Chip` chips.
fn hbm_window<D: Scalar, Chip: M>(base_len: usize, window_base: usize) -> std::ops::Range<usize> {
    assert_ne!(Chip::SIZE, 0, "an HBM view must span at least one chip");
    assert!(
        base_len.is_multiple_of(Chip::SIZE),
        "HBM view elements must divide evenly across chips"
    );
    let elements = base_len / Chip::SIZE;
    assert_ne!(elements, 0, "an HBM view must span at least one element per chip");
    D::size_in_bytes_from_length(window_base % elements)..D::size_in_bytes_from_length(elements)
}

/// Bytes per chip of a `m![Chip, Element]` tensor of `D` at `offset` in `buffer`; fails unless it
/// spans every chip, ends inside the buffer and is aligned to `D`.
fn tensor_bytes<D: Scalar, Chip: M, Element: M>(buffer: &impl Allocation, offset: usize) -> Result<usize, Error> {
    let bytes = Element::SIZE
        .checked_mul(D::BITS)
        .filter(|bits| bits.is_multiple_of(8))
        .map(|bits| bits / 8)
        .ok_or(Error::Invalid("the per-chip tensor extent is not a whole byte count"))?;
    if Chip::SIZE != buffer.chips() {
        return Err(Error::Invalid("the chip extent does not match the buffer"));
    }
    if offset.checked_add(bytes).is_none_or(|end| end > buffer.size()) {
        return Err(Error::Invalid("the tensor extends past the buffer"));
    }
    if !(buffer.addr() + offset).is_multiple_of((D::BITS / 8).max(1)) {
        return Err(Error::Invalid("the tensor is not aligned to its element size"));
    }
    Ok(bytes)
}

/// The bytes of `region` that `D` values laid out as `m![Chip, Element]` take: every chip of
/// `buffer`, from the region's start.
fn fit_slice<D: Scalar, Chip: M, Element: M>(buffer: &impl Allocation, region: &Region) -> Result<Region, Error> {
    if region.chips.count_ones() as usize != buffer.chips() {
        return Err(Error::Invalid("a tensor view must span every chip of the buffer"));
    }
    let end = region.bytes.start + tensor_bytes::<D, Chip, Element>(buffer, region.bytes.start)?;
    if end > region.bytes.end {
        return Err(Error::Invalid("the tensor extends past the slice"));
    }
    Ok(Region {
        chips: region.chips,
        bytes: region.bytes.start..end,
    })
}

/// A slice's bytes read as `D` values laid out as `m![Chip, Element]`.
impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> TryFrom<HbmBufferSlice<'l, B>>
    for HbmTensorView<'l, D, Chip, Element, B>
{
    type Error = Error;

    fn try_from(slice: HbmBufferSlice<'l, B>) -> Result<Self, Error> {
        let region = fit_slice::<D, Chip, Element>(slice.buffer, &slice.region)?;
        Ok(Self {
            inner: View::whole(HbmBufferSlice {
                buffer: slice.buffer,
                region,
            }),
            chips: PhantomData,
        })
    }
}

/// [`HbmTensorView`]'s conversion, to write.
impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> TryFrom<HbmBufferSliceMut<'l, B>>
    for HbmTensorViewMut<'l, D, Chip, Element, B>
{
    type Error = Error;

    fn try_from(slice: HbmBufferSliceMut<'l, B>) -> Result<Self, Error> {
        let region = fit_slice::<D, Chip, Element>(&*slice.buffer, &slice.region)?;
        Ok(Self {
            inner: View::whole(HbmBufferSliceMut {
                buffer: slice.buffer,
                region,
            }),
            chips: PhantomData,
        })
    }
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B, AllChips> {
    /// The device memory this view covers: its tile of the base tensor.
    pub(crate) fn memory(&self) -> HbmBufferSlice<'_, B> {
        HbmBufferSlice {
            buffer: self.inner.inner.buffer,
            region: Region {
                chips: self.inner.inner.region.chips,
                bytes: self.window(),
            },
        }
    }

    /// The per-chip byte range of the base allocation this view covers.
    pub(crate) fn window(&self) -> std::ops::Range<usize> {
        let start = self.inner.inner.region.bytes.start;
        let window = hbm_window::<D, Chip>(self.inner.window.base_len(), self.inner.window.window_base());
        start + window.start..start + window.end
    }
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend, Chips> HbmTensorView<'l, D, Chip, Element, B, Chips> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Element }];

    /// Runs `read` over the view's cells, as a device function reads them.
    fn read<const DMA: Dma, R>(
        &self,
        dma: &DmaContext<DMA, B>,
        read: impl FnOnce(TensorView<'_, D, Pair<Chip, Element>, B>) -> R,
    ) -> R {
        let base = dma.load::<D>(self.inner.inner.clone());
        read(self.inner.window.over::<D, B>(&base))
    }

    /// Writes to HBM tensor view. The destination's `Chip2` is free of the
    /// source's `Chip`: a read source (Top padding) may target a `view_mut`
    /// destination (Bottom padding). `transpose` validates the live layout.
    #[primitive(HbmTensorView::to_hbm_view)]
    pub fn to_hbm_view<const DMA: Dma, Chip2: M, Element2: M, Chips2>(
        self,
        dma: &mut DmaContext<{ DMA }, B>,
        dst: HbmTensorViewMut<'l, D, Chip2, Element2, B, Chips2>,
    ) {
        self.read(dma, |src| dst.write(dma, |mut dst| dst.transpose(src, true)));
    }

    /// Writes to data memory tensor view.
    #[primitive(HbmTensorView::to_dm_view)]
    pub fn to_dm_view<Chip2: M, Cluster: M, Slice: M, Element2: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element2, B>,
    ) {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip2 }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        self.read(dma, |src| dst.inner.transpose(src, true));
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(HbmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> HbmTensorView<'l, D, Chip2, Element, B, SomeChips> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        HbmTensorView {
            inner: View {
                inner: self.inner.inner.clone(),
                window: self
                    .inner
                    .window
                    .clone()
                    .retile::<Index, _>(start, LEN, PaddingKind::Top),
                marker: PhantomData,
            },
            chips: PhantomData,
        }
    }

    /// Creates immutable views by splitting along a tile expression.
    #[primitive(HbmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> HbmTensorView<'l, D, Chip, Element2, B, Chips> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        HbmTensorView {
            inner: View {
                inner: self.inner.inner.clone(),
                window: self
                    .inner
                    .window
                    .clone()
                    .retile::<Index, _>(start, LEN, PaddingKind::Top),
                marker: PhantomData,
            },
            chips: PhantomData,
        }
    }

    /// Reshapes the view at the same HBM address, consuming `self`. A reshape is a MOVE: a zero-copy
    /// rewrap of the same borrow (see [`TensorView::reshape`]). Sound on every backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Element` sizes asserted at compile time (see
    /// [`constraints::assert_hbm_reshape_dimension_preserved`]); precondition is same-wire-order
    /// (regroup valid, permutation not). Consuming `self` makes the move explicit.
    #[primitive(HbmTensorView::reshape)]
    pub unsafe fn reshape<Chip2: M, Element2: M>(self) -> HbmTensorView<'l, D, Chip2, Element2, B, Chips> {
        constraints::assert_hbm_reshape_dimension_preserved::<Chip, Chip2, Element, Element2>();
        HbmTensorView {
            inner: View {
                inner: self.inner.inner,
                window: self.inner.window.reshape::<m![{ Chip2 }, { Element2 }]>(),
                marker: PhantomData,
            },
            chips: PhantomData,
        }
    }

    /// Views the same cells inside the wider buffer they sit in, restating `Element` with one
    /// outermost padding factor: `m![L1]` as `m![L1 # 256]`. See [`DmTensorView::pad`]; on an HBM
    /// SOURCE this is how a DMA states the extent its destination is staged at, so the two sides carve
    /// the same tail. The read over-reaches the live cells into declared don't-care, which is what a
    /// read packet is allowed to do -- `Pad::extent` trims to the live cells on the SINK side only.
    #[primitive(HbmTensorView::pad)]
    pub fn pad<Element2: M>(self) -> HbmTensorView<'l, D, Chip, Element2, B, Chips> {
        config_pad(PadInput {
            element: Element::to_value(),
            expected: Element2::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        HbmTensorView {
            inner: View {
                inner: self.inner.inner,
                window: self.inner.window.redeclare::<m![{ Chip }, { Element2 }]>(),
                marker: PhantomData,
            },
            chips: PhantomData,
        }
    }
}

impl<'l, D: MaterializableScalar, Chip: M, Element: M, B: Backend, Chips>
    HbmTensorView<'l, D, Chip, Element, B, Chips>
{
    /// Converts to data memory tensor.
    #[primitive(HbmTensorView::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        self.read(dma, |src| DmTensor::from_parts(src.read().transpose(true)))
    }

    /// Redistributes HBM chip slots by DMA: `shuffle_pattern[target] = source`.
    /// Panics unless the pattern is a permutation of every position in `Chip`.
    #[primitive(HbmTensorView::hbm_chip_shuffle)]
    pub fn hbm_chip_shuffle<const CHIP_DIM: usize, const DMA: Dma>(
        self,
        dma: &mut DmaContext<{ DMA }, B>,
        shuffle_pattern: &[usize; CHIP_DIM],
    ) -> HbmTensor<D, Chip, Element, B> {
        assert_chip_shuffle_pattern(shuffle_pattern, Chip::SIZE);
        let mut shuffled: HbmTensor<D, Chip, Element, B> = HbmTensor::new();

        for (target_chip_idx, source_chip_idx) in shuffle_pattern.iter().enumerate() {
            self.chip_tile::<Chip, 1, Padding<Identity, Broadcast<CHIP_DIM>>>(*source_chip_idx)
                .to_hbm_view(
                    dma,
                    shuffled
                        .view_mut()
                        .chip_tile::<Chip, 1, Padding<Identity, Broadcast<CHIP_DIM>, { PaddingKind::Bottom }>>(
                            target_chip_idx,
                        ),
                );
        }

        shuffled
    }
}

/// Mutable view of an HBM tensor; see [`HbmTensorView`] for `Chips`.
#[primitive(HbmTensorViewMut)]
#[derive(Debug)]
pub struct HbmTensorViewMut<'l, D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend, Chips = AllChips> {
    /// The view's cells, over the base tensor's device memory.
    inner: View<'l, D, Pair<Chip, Element>, B, HbmBufferSliceMut<'l, B>>,
    chips: PhantomData<Chips>,
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorViewMut<'l, D, Chip, Element, B, AllChips> {
    /// The device memory this view covers: its tile of the base tensor.
    pub(crate) fn memory(&self) -> HbmBufferSlice<'_, B> {
        HbmBufferSlice {
            buffer: &*self.inner.inner.buffer,
            region: Region {
                chips: self.inner.inner.region.chips,
                bytes: self.window(),
            },
        }
    }

    /// The per-chip byte range of the base allocation this view covers.
    pub(crate) fn window(&self) -> std::ops::Range<usize> {
        let start = self.inner.inner.region.bytes.start;
        let window = hbm_window::<D, Chip>(self.inner.window.base_len(), self.inner.window.window_base());
        start + window.start..start + window.end
    }
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend, Chips> HbmTensorViewMut<'l, D, Chip, Element, B, Chips> {
    /// Runs `write` over the view's cells, as a device function writes them. The cells return to
    /// memory even when `write` unwinds, so a caught panic leaves the memory whole, as on a device.
    pub(crate) fn write<const DMA: Dma>(
        mut self,
        dma: &DmaContext<DMA, B>,
        write: impl FnOnce(TensorViewMut<'_, D, Pair<Chip, Element>, B>),
    ) {
        struct Taken<'a, 'l, const DMA: Dma, D: Scalar, B: Backend> {
            dma: &'a DmaContext<DMA, B>,
            taken_memory: &'a mut HbmBufferSliceMut<'l, B>,
            base: B::Storage<D>,
        }
        impl<const DMA: Dma, D: Scalar, B: Backend> Drop for Taken<'_, '_, DMA, D, B> {
            fn drop(&mut self) {
                self.dma.store(
                    HbmBufferSliceMut {
                        buffer: &mut *self.taken_memory.buffer,
                        region: self.taken_memory.region.clone(),
                    },
                    &self.base,
                );
            }
        }
        let base = dma.load::<D>(self.inner.inner.as_ref());
        let mut taken = Taken {
            dma,
            taken_memory: &mut self.inner.inner,
            base,
        };
        write(self.inner.window.over_mut::<D, B>(&mut taken.base));
    }

    /// Creates mutable views by splitting along a tile expression over Chip.
    #[primitive(HbmTensorViewMut::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip2, Element, B, SomeChips> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        HbmTensorViewMut {
            inner: View {
                inner: self.inner.inner,
                window: self.inner.window.retile::<Index, _>(start, LEN, PaddingKind::Bottom),
                marker: PhantomData,
            },
            chips: PhantomData,
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(HbmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        self,
        start: usize,
    ) -> HbmTensorViewMut<'l, D, Chip, Element2, B, Chips> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        HbmTensorViewMut {
            inner: View {
                inner: self.inner.inner,
                window: self.inner.window.retile::<Index, _>(start, LEN, PaddingKind::Bottom),
                marker: PhantomData,
            },
            chips: PhantomData,
        }
    }

    /// Reshapes the mutable view at the same HBM address, consuming `self`. A reshape is a MOVE: a
    /// zero-copy rewrap of the same `&mut` borrow (see [`TensorViewMut::reshape`]). Sound on every
    /// backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Element` sizes asserted at compile time (see
    /// [`constraints::assert_hbm_reshape_dimension_preserved`]); precondition is same-wire-order
    /// (regroup valid, permutation not). Consuming `self` makes the move explicit.
    #[primitive(HbmTensorViewMut::reshape)]
    pub unsafe fn reshape<Chip2: M, Element2: M>(self) -> HbmTensorViewMut<'l, D, Chip2, Element2, B, Chips> {
        constraints::assert_hbm_reshape_dimension_preserved::<Chip, Chip2, Element, Element2>();
        HbmTensorViewMut {
            inner: View {
                inner: self.inner.inner,
                window: self.inner.window.reshape::<m![{ Chip2 }, { Element2 }]>(),
                marker: PhantomData,
            },
            chips: PhantomData,
        }
    }
}

/// The host backend's device memory is host memory, so its tensors read synchronously; on a device,
/// HBM bytes move only by DMA ([`HbmTensor::to_host`]).
impl<D: MaterializableScalar, Chip: M, Element: M> HbmTensor<D, Chip, Element, Cpu> {
    pub(crate) fn load(&self) -> Tensor<D, Self::Mapping, Cpu> {
        let memory = self.memory();
        Tensor::from_inner(memory.buffer.load(&memory.region).into())
    }

    /// The dense packed byte image in `m![Chip, Element]` order, as the device holds it: a 4-bit
    /// scalar packs two codes per byte.
    pub fn to_buf(&self) -> Vec<u8> {
        self.load().into_buf()
    }
}

impl<D: MaterializableScalar, Chip: M, Element: M, Chips> HbmTensorView<'_, D, Chip, Element, Cpu, Chips> {
    /// The dense packed byte image of the view's cells; see [`HbmTensor::to_buf`].
    pub fn to_buf(&self) -> Vec<u8> {
        let memory = &self.inner.inner;
        let base = memory.buffer.load(&memory.region).into();
        self.inner.window.over::<D, Cpu>(&base).read().into_buf()
    }
}

impl<D: MaterializableScalar, Chip: M, Element: M, Chips> HbmTensorViewMut<'_, D, Chip, Element, Cpu, Chips> {
    /// The dense packed byte image of the view's cells; see [`HbmTensor::to_buf`].
    pub fn to_buf(&self) -> Vec<u8> {
        let memory = &self.inner.inner;
        let base = memory.buffer.load(&memory.region).into();
        self.inner.window.over::<D, Cpu>(&base).read().into_buf()
    }
}

/// Tensor stored in scratchpad memory. `Pe` places the payload across the cluster's fused PEs.
#[primitive(SpmTensor)]
#[derive(Debug)]
pub struct SpmTensor<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Pe, Element>>>, B>,
    _marker: PhantomData<(D, Chip, Cluster, Pe, Element)>,
}

impl<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend> SpmTensor<D, Chip, Cluster, Pe, Element, B> {
    /// Logical SPM mapping, including the explicit fused PE axis.
    pub type Mapping = m![{ Chip }, { Cluster }, { Pe }, { Element }];

    /// Constructs an SPM tensor after checking, at compile time, that `Cluster` and `Pe` name
    /// hardware counts and that `Element` fits the scratchpad one PE holds. The over-budget case is
    /// a `compile_fail` doctest in `furiosa-opt-examples`' `negative::scatter_gather`.
    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>) -> Self {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_pe_size::<Pe>();
        constraints::assert_spm_capacity::<D, Element>();
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Converts to a fresh DM tensor.
    #[primitive(SpmTensor::to_dm)]
    pub fn to_dm<Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2, B> {
        constraints::assert_pe_matches_slice::<Pe, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Pe }, { Element }],
            Element,
            m![{ Cluster }, { Slice2 }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        DmTensor::from_parts(self.inner.transpose(true))
    }
}

/// Tensor stored in data memory.
#[primitive(DmTensor)]
#[derive(Debug)]
pub struct DmTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
    // The resident SRAM allocation this tensor names and keeps live; `None` for a tensor the
    // compiled program places itself.
    owner: Option<B::Buffer>,
}

/// Allocation failure for a runtime-owned DM tensor.
#[derive(thiserror::Error, Clone, Debug, Eq, PartialEq)]
pub enum DmAllocError {
    /// The tensor and device must cover the same chips.
    #[error("the tensor spans {tensor_chips} chips, but the device has {device_chips}")]
    Topology {
        /// Chips in the tensor mapping.
        tensor_chips: usize,
        /// Chips opened by the device.
        device_chips: usize,
    },
    /// The runtime allocation failure, preserving its structured cause.
    #[error(transparent)]
    Allocation(#[from] furiosa_opt_rt::Error),
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// `Cluster` / `Slice` map to physical SRAM partitions; checked in `from_parts` so every DM
    /// tensor constructor validates them at compile time. One `const` block per check so a bad
    /// `Cluster` and a bad `Slice` each report their own error (a single block stops at the first
    /// panic).
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    pub(crate) fn from_parts(inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self { inner, owner: None }
    }

    /// Names the resident allocation this tensor lives in and keeps it live for the tensor's
    /// lifetime.
    pub(crate) fn own(&mut self, owner: B::Buffer) {
        self.owner = Some(owner);
    }

    /// The resident allocation this tensor is placed in, or `None` while the compiled program
    /// owns its placement.
    pub(crate) fn owner(&self) -> Option<&B::Buffer> {
        self.owner.as_ref()
    }

    /// Allocates uninitialized SRAM on `ctx`, kept alive by this tensor and its launches.
    /// Initialize it before reading; pass it by reference to kernels on the same device group.
    pub fn alloc(ctx: &crate::Device<B>) -> Result<Self, DmAllocError> {
        B::alloc_dm(ctx.pdma.device())
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Creates a fresh DM tensor handle with no assigned address. The backend places it.
    ///
    /// `Cluster` / `Slice` are validated at compile time (see [`Self::from_parts`]). A bad partition
    /// is rejected before codegen; because each check sits in its own `const` block, every violated
    /// check reports its own error (here `Cluster = 3` and `Slice = 5` both do, in one compile):
    ///
    /// ```compile_fail
    /// use furiosa_opt_std::prelude::*;
    /// // Cluster must be 1 | 2 and Slice must be 64 | 128 | 256.
    /// let _ = DmTensor::<i32, m![1], m![3], m![5], m![8]>::new();
    /// ```
    // `new()` builds an uninitialized handle, so a `Default` impl (which would look zero-cost and
    // safe) is deliberately not provided.
    #[allow(clippy::new_without_default)]
    #[primitive(DmTensor::new)]
    pub fn new() -> Self {
        Self::from_parts(Tensor::zeroed())
    }
}

/// Determines which hardware dimension keys an asymmetric slice's indices.
#[derive(Clone, Copy)]
enum SliceIndexing {
    PerChip,
    PerCluster,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Creates immutable views by splitting along a tile expression.
    #[primitive(DmTensor::view)]
    pub fn view<'l>(&'l self) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element, B> {
        DmTensorView {
            inner: self.inner.view(),
        }
    }

    /// Creates mutable views by splitting along a tile expression.
    #[primitive(DmTensor::view_mut)]
    pub fn view_mut<'l>(&'l mut self) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.view_mut(),
        }
    }

    /// Selects one live position of an outermost, 8-byte-aligned axis per chip.
    #[primitive(DmTensor::asymmetric_chip_slice)]
    pub fn asymmetric_chip_slice<AxisToSlice: M, Element2: M>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
        slice_indices: &[usize],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        self.asymmetric_slice::<AxisToSlice, Element2>(sub, slice_indices, SliceIndexing::PerChip)
    }

    /// Selects one live position of an outermost, 8-byte-aligned axis per cluster.
    #[primitive(DmTensor::asymmetric_cluster_slice)]
    pub fn asymmetric_cluster_slice<AxisToSlice: M, Element2: M>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
        slice_indices: &[usize],
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        self.asymmetric_slice::<AxisToSlice, Element2>(sub, slice_indices, SliceIndexing::PerCluster)
    }

    fn asymmetric_slice<AxisToSlice: M, Element2: M>(
        &self,
        _sub: &mut TuContext<{ Tu::Sub }>,
        slice_indices: &[usize],
        indexing: SliceIndexing,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_asymmetric_slice::<D, AxisToSlice, Element, Element2>();
        let targets = match indexing {
            SliceIndexing::PerChip => Chip::SIZE,
            SliceIndexing::PerCluster => Cluster::SIZE,
        };
        assert_slice_indices::<AxisToSlice>(slice_indices, targets);
        let tensor = self.view();
        let mut sliced = DmTensor::new();

        for (target, slice_idx) in slice_indices.iter().enumerate() {
            match indexing {
                SliceIndexing::PerChip => {
                    let selected = tensor
                        .chip_tile_derived::<Chip, 1>(target)
                        .slice_axis::<AxisToSlice>(*slice_idx);
                    sliced
                        .view_mut()
                        .chip_tile_derived::<Chip, 1>(target)
                        .inner
                        .transpose(selected.inner, false);
                }
                SliceIndexing::PerCluster => {
                    let selected = tensor
                        .cluster_tile_derived::<Cluster, 1>(target)
                        .slice_axis::<AxisToSlice>(*slice_idx);
                    sliced
                        .view_mut()
                        .cluster_tile_derived::<Cluster, 1>(target)
                        .inner
                        .transpose(selected.inner, false);
                }
            }
        }

        sliced
    }

    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(DmTensor::to_hbm)]
    pub fn to_hbm<Element2: M>(&self, dma: &mut DmaContext<{ Dma::Tensor }, B>) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::from_value(dma, &self.inner.transpose(true))
    }

    /// Converts to a fresh SPM tensor with the caller-provided PE and element mappings.
    #[primitive(DmTensor::to_spm)]
    pub fn to_spm<Pe2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> SpmTensor<D, Chip, Cluster, Pe2, Element2, B> {
        assert_spm_staging::<D, Cluster, Slice, Element, Pe2, Element2>();
        SpmTensor::new(self.inner.transpose(true))
    }

    /// Converts to a data-memory tensor with the requested dimension mappings.
    #[primitive(DmTensor::to_dm)]
    pub fn to_dm<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        assert_dm_dma_layout::<D, Chip, Cluster, Slice, Element, Chip2, Cluster2, Slice2, Element2>(
            DM_WRITE_ALIGN_BYTES,
        );
        DmTensor::from_parts(self.inner.transpose(true))
    }

    /// Reshapes the tensor to a different mapping at the same address, consuming `self`. Delegates to
    /// [`Tensor::reshape`].
    ///
    /// # Safety
    ///
    /// The per-level sizes (`Chip::SIZE == Chip2::SIZE`, `Cluster`, `Slice`, `Element`) are asserted
    /// below; the genuine precondition is [`Tensor::reshape`]'s: the old and new DM mappings must lay
    /// the elements out in the SAME physical (wire) order, so the relabel moves no data. Axis
    /// regrouping is valid; a permutation is not (use a transpose). Equal sizes do not guarantee this.
    /// Consuming `self` makes the move explicit: reshape is a MOVE (neither alias nor copy), so no
    /// old-shaped handle survives to alias the same bytes under a conflicting mapping.
    #[primitive(DmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        let reshaped = unsafe {
            self.inner
                .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
        };
        let mut reshaped = DmTensor::from_parts(reshaped);
        reshaped.owner = self.owner;
        reshaped
    }
}

/// Mutable view of a data memory tensor.
#[primitive(DmTensorViewMut)]
#[derive(Debug)]
pub struct DmTensorViewMut<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: TensorViewMut<'l, D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
}

/// View of a data memory tensor.
#[primitive(DmTensorView)]
#[derive(Debug, Clone)]
pub struct DmTensorView<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: TensorView<'l, D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    From<DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>>
    for DmTensorView<'l, D, Chip, Cluster, Slice, Element, B>
{
    fn from(view: DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>) -> Self {
        Self {
            inner: view.inner.into(),
        }
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorView<'l, D, Chip, Cluster, Slice, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// Writes data to a mutable tensor view for HBM. `Chip2` is free of the
    /// source's `Chip`: a read source (Top) may target a `view_mut` destination
    /// (Bottom). `transpose` validates the live layout.
    #[primitive(DmTensorView::to_hbm_view)]
    pub fn to_hbm_view<Chip2: M, Element2: M, Chips>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        dst: HbmTensorViewMut<'l, D, Chip2, Element2, B, Chips>,
    ) {
        dst.write(dma, |mut dst| dst.transpose(self.inner, true));
    }

    /// Writes data to a mutable tensor view for data memory. `Chip2`/`Cluster2`
    /// are free of the source's: a read source (Top) may target a `view_mut`
    /// destination (Bottom). `transpose` validates the live layout.
    #[primitive(DmTensorView::to_dm_view)]
    pub fn to_dm_view<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster2, Slice2, Element2, B>,
    ) {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Slice }, { Element }],
            Element,
            m![{ Cluster2 }, { Slice2 }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        dst.inner.transpose(self.inner, true);
    }

    /// Copies this DM view into a fresh SPM tensor with the requested PE and element mappings.
    #[primitive(DmTensorView::to_spm)]
    pub fn to_spm<Pe2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> SpmTensor<D, Chip, Cluster, Pe2, Element2, B> {
        assert_spm_staging::<D, Cluster, Slice, Element, Pe2, Element2>();
        let mut output = SpmTensor::new(Tensor::zeroed());
        output.inner.view_mut().transpose(self.inner, true);
        output
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(DmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip2, Cluster, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorView { inner }
    }

    pub(crate) fn chip_tile_derived<Index: M, const LEN: usize>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Tiled<Index, Chip, LEN, { PaddingKind::Top }>, Cluster, Slice, Element, B> {
        DmTensorView {
            inner: self.inner.retile_derived::<Index, _>(start, LEN),
        }
    }

    pub(crate) fn slice_axis<AxisToSlice: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, Tiled<AxisToSlice, Element, 1, { PaddingKind::Top }>, B> {
        DmTensorView {
            inner: self.inner.retile::<AxisToSlice, _>(start, 1),
        }
    }

    /// Creates immutable views by splitting along a tile expression over Cluster.
    #[primitive(DmTensorView::cluster_tile)]
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster2, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster2, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Cluster::to_value(),
            expected: Cluster2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorView { inner }
    }

    pub(crate) fn cluster_tile_derived<Index: M, const LEN: usize>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Tiled<Index, Cluster, LEN, { PaddingKind::Top }>, Slice, Element, B> {
        DmTensorView {
            inner: self.inner.retile_derived::<Index, _>(start, LEN),
        }
    }

    /// Creates immutable views by splitting along a tile expression over Slice.
    #[primitive(DmTensorView::slice_tile)]
    pub fn slice_tile<Index: M, const LEN: usize, Slice2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice2, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Slice::to_value(),
            expected: Slice2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorView { inner }
    }

    /// Reshapes the view over the same borrow, consuming `self`. A reshape is a MOVE: a zero-copy
    /// rewrap of the same borrow (see [`TensorView::reshape`]). Sound on every backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Cluster`/`Slice` asserted at compile time, matching [`DmTensor::reshape`] (see
    /// [`constraints::assert_reshape_dimension_preserved`] -- `Element` isn't checked there either,
    /// per its own TODO: some current examples reshape with a mismatched `Element`). Precondition is
    /// same-wire-order (regroup within `Element` valid; a permutation is not, use a transpose).
    #[primitive(DmTensorView::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensorView<'l, D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        DmTensorView {
            inner: unsafe {
                self.inner
                    .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
            },
        }
    }

    /// Views the same cells inside the wider buffer they sit in, restating `Element` with one
    /// outermost padding factor: `m![L4, B % 64]` as `m![L4 # 512, B % 64]`. The inverse of the
    /// [`Self::tile`] that reads a padded tensor's live rows (see [`TensorView::pad`] for why it is
    /// safe), and the way to give an interleave's two operands one `Element` without narrowing the
    /// wider one and losing its extent.
    #[primitive(DmTensorView::pad)]
    pub fn pad<Element2: M>(self) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        // `Element` alone: the buffer a pad re-declares is the per-slice region, so the padding is
        // outermost within `Element`. The distribution classes ride through untouched.
        config_pad(PadInput {
            element: Element::to_value(),
            expected: Element2::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        DmTensorView {
            inner: self
                .inner
                .redeclare::<m![{ Chip }, { Cluster }, { Slice }, { Element2 }]>(),
        }
    }

    /// Views a wider-staged producer's live cells alone, dropping one outermost padding factor from
    /// `Element`: `m![L1 # 256, B % 64]` as `m![L1, B % 64]`. The inverse of [`Self::pad`]; see
    /// [`TensorView::unpad`] for why it is address-preserving and why a `tile` is not a substitute.
    #[primitive(DmTensorView::unpad)]
    pub fn unpad<Element2: M>(self) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        config_pad(PadInput {
            element: Element2::to_value(),
            expected: Element::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        DmTensorView {
            inner: self
                .inner
                .redeclare::<m![{ Chip }, { Cluster }, { Slice }, { Element2 }]>(),
        }
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element, B>
{
    fn reborrow(&mut self) -> DmTensorViewMut<'_, D, Chip, Cluster, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.reborrow(),
        }
    }

    /// Creates mutable views by splitting along a tile expression over Chip.
    #[primitive(DmTensorViewMut::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Chip::to_value(),
            expected: Chip2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorViewMut { inner }
    }

    pub(crate) fn chip_tile_derived<Index: M, const LEN: usize>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Tiled<Index, Chip, LEN, { PaddingKind::Bottom }>, Cluster, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.retile_derived::<Index, _>(start, LEN),
        }
    }

    /// Creates mutable views by splitting along a tile expression over Cluster.
    #[primitive(DmTensorViewMut::cluster_tile)]
    pub fn cluster_tile<Index: M, const LEN: usize, Cluster2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster2, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster2, Slice, Slice>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Cluster::to_value(),
            expected: Cluster2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorViewMut { inner }
    }

    pub(crate) fn cluster_tile_derived<Index: M, const LEN: usize>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Tiled<Index, Cluster, LEN, { PaddingKind::Bottom }>, Slice, Element, B> {
        DmTensorViewMut {
            inner: self.inner.retile_derived::<Index, _>(start, LEN),
        }
    }

    /// Creates mutable views by splitting along a tile expression over Slice.
    #[primitive(DmTensorViewMut::slice_tile)]
    pub fn slice_tile<Index: M, const LEN: usize, Slice2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice2, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip, Cluster, Cluster, Slice, Slice2>();
        config_tile(TileInput {
            index: Index::to_value(),
            element: Slice::to_value(),
            expected: Slice2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2, B> {
        config_tile(TileInput {
            index: Index::to_value(),
            element: Element::to_value(),
            expected: Element2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorViewMut { inner }
    }

    /// Reshapes the mutable view over the same `&mut` borrow, consuming `self`. A reshape is a MOVE: a
    /// zero-copy rewrap of the same borrow (see [`TensorViewMut::reshape`]). Sound on every backend.
    ///
    /// # Safety
    ///
    /// `Chip`/`Cluster`/`Slice` asserted at compile time, matching [`DmTensor::reshape`] (see
    /// [`constraints::assert_reshape_dimension_preserved`] -- `Element` isn't checked there either,
    /// per its own TODO: some current examples reshape with a mismatched `Element`). Precondition is
    /// same-wire-order (regroup within `Element` valid; a permutation is not, use a transpose).
    #[primitive(DmTensorViewMut::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> DmTensorViewMut<'l, D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        DmTensorViewMut {
            inner: unsafe {
                self.inner
                    .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
            },
        }
    }

    /// Fills this view's region with the typed value-domain `value` (`bf16::from_f32(1.0)`, not a
    /// `0x3f80` bit pattern), lowering to one on-device fill command.
    ///
    /// `value` must be compile-time constant so it const-folds to the fill's element bits: a plain
    /// literal folds directly, a computed value needs a `const` block, e.g.
    /// `memset(const { bf16::from_f32(1.0) }, ..)`. A non-constant `value` is rejected at translation.
    ///
    /// Only a whole-region fill (a bare `view_mut()`) is supported today; a sub-view fill
    /// (`view_mut().tile(..).memset(..)`) is rejected at translation, pending a ranged fill. Supports
    /// every [`Scalar`] this branch translates to vISA: the byte+
    /// RNGD scalars (`i8`, `i16`, `i32`, `f32`, `bf16`, `f8e4m3`) and sub-byte `i4` (fill materialized
    /// from the low `D::BITS`; a sub-byte region must be byte-aligned or it is rejected). `f4e2m1` is a
    /// follow-up: a valid DSL `Scalar`, but its vISA support lands with the separate
    /// fetch/table-lookup work, so a `memset(f4e2m1_value)` is rejected rather than mis-lowered.
    #[primitive(DmTensorViewMut::memset)]
    pub fn memset(&mut self, value: D, _sub: &mut TuContext<{ Tu::Sub }>) {
        // Reference backends fill this view's live cells (padding stays untouched, matching the device
        // write-back); the device path emits the fill command into the viewed region.
        let fill: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B> = Tensor::splat(value);
        self.inner.transpose(fill.view(), true);
    }
}

// ANCHOR: trf_tensor_def
/// Tensor stored in the tensor register file.
#[primitive(TrfTensor)]
#[derive(Debug)]
pub struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Element>>>>, B>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Lane, Element)>,
}
// ANCHOR_END: trf_tensor_def

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Element }];

    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// A fresh TRF tensor, zero-filled. Where in the register file it lands is the compiler's to
    /// decide, so the handle carries no address of its own.
    pub fn zero() -> Self {
        Self::new(Tensor::zeroed())
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M, B: Backend>
    TrfTensor<D, Chip, Cluster, Slice, Lane, Element, B>
{
    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping, B> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping, B> {
        self.inner.view()
    }
}

// ANCHOR: vrf_tensor_def
/// Tensor stored in the vector register file (VRF).
#[primitive(VrfTensor)]
#[derive(Debug, Clone)]
pub struct VrfTensor<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
    _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
// ANCHOR_END: vrf_tensor_def

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Element }];

    /// Every VRF handle is built here, which is where one slice's `Element` answers to the file it
    /// has to fit.
    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>) -> Self {
        constraints::assert_vrf_capacity::<D, Element>();
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// A fresh VRF tensor, zero-filled. Where in the register file it lands is the compiler's to
    /// decide, so the handle carries no address of its own.
    pub fn zero() -> Self {
        Self::new(Tensor::zeroed())
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    VrfTensor<D, Chip, Cluster, Slice, Element, B>
{
    /// Restates the operand under a different mapping over the same register, consuming `self`. Its
    /// use is naming the slices a broadcast operand already holds a copy in, so it can feed a stream
    /// partitioned by a named axis (the operand rule is
    /// [`IntoBranchedOperand`](crate::engine::vector::operand::IntoBranchedOperand)'s).
    ///
    /// The VRF peer of [`DmTensor::reshape`]; both delegate to [`Tensor::reshape`] and relabel rather
    /// than relayout, so no element moves.
    ///
    /// # Safety
    ///
    /// Every physical position (chip, cluster, slice, in-slice element) must already hold what the
    /// new mapping claims of it. Two forms do that, and this call must be one of them:
    ///
    /// 1. **Regrouping axes**, `m![W]` -> `m![W / 2, W % 2]`. This is [`Tensor::reshape`]'s own
    ///    precondition: both mappings lay the SAME live elements out in the SAME wire order.
    /// 2. **Naming a broadcast distribution axis**, `m![256]` -> `m![W]`. Here the live elements do
    ///    NOT line up, since a `Broadcast<256>` carries one of them and `m![W]` carries 256. It is
    ///    sound for the other reason: a broadcast at `Chip` / `Cluster` / `Slice` means those units
    ///    each hold the same copy, so naming the copies reads no other unit's data. `Element` has no
    ///    such reading, and a broadcast one relabelled to a named axis invents data.
    ///
    /// Permuting axes (`m![A, B]` -> `m![B, A]`) is neither; that is [`Tensor::transpose`]'s job.
    ///
    /// Only per-level `SIZE` equality is asserted, so which of the two forms this is stays the
    /// caller's to know. A violation is not UB but silence: the backend relabel is a volume
    /// `assert_eq!` plus a copy, so it yields wrong values under emulation and a wrong EDF once
    /// compiled, with no diagnostic on either path.
    #[primitive(VrfTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
    ) -> VrfTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        constraints::assert_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Slice,
            Slice2,
            Element,
            Element2,
        >();
        let reshaped = unsafe {
            self.inner
                .reshape::<m![{ Chip2 }, { Cluster2 }, { Slice2 }, { Element2 }]>()
        };
        VrfTensor::new(reshaped)
    }

    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping, B> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping, B> {
        self.inner.view()
    }
}

/// Tensor stored in dot product engine
#[derive(Debug)]
pub struct DpeTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Lane: M, Packet: M, B: Backend = CurrentBackend>
{
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Pair<Lane, Packet>>>>>, B>,
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Lane: M, Packet: M, B: Backend>
    DpeTensor<D, Chip, Cluster, Slice, Time, Lane, Packet, B>
{
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Time }, { Lane }, { Packet }];
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Lane: M, Packet: M, B: Backend>
    DpeTensor<D, Chip, Cluster, Slice, Time, Lane, Packet, B>
{
    /// Creates a mutable view into the tensor.
    pub fn view_mut<'l>(&'l mut self) -> TensorViewMut<'l, D, Self::Mapping, B> {
        self.inner.view_mut()
    }

    /// Creates an immutable view into the tensor.
    pub fn view<'l>(&'l self) -> TensorView<'l, D, Self::Mapping, B> {
        self.inner.view()
    }
}

fn assert_spm_staging<D: Scalar, Cluster: M, Slice: M, Element: M, Pe2: M, Element2: M>() {
    constraints::assert_pe_matches_slice::<Pe2, Slice>();
    // The hardware config treats SPM as non-SRAM, so staging uses DRAM access width 1.
    assert_dma_layout::<
        D,
        m![{ Cluster }, { Slice }, { Element }],
        Element,
        m![{ Cluster }, { Pe2 }, { Element2 }],
        Element2,
    >(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::Cpu;
    use crate::runtime::{Device, Topology};

    /// A whole HBM view uses the reshaped axes for subsequent reads.
    #[test]
    fn reads_reshaped_cells() {
        axes![X = 8, A = 4, B = 2];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let tensor = HbmTensor::<i32, m![1], m![X], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..8));
        // SAFETY: eight consecutive cells become four consecutive pairs, with no data movement.
        let view = unsafe { tensor.view().reshape::<m![1], m![A, B]>() };
        let got = view
            .tile::<m![A], 1, m![A = 1 # 4, B]>(1)
            .read(&device.tdma, |cells| cells.read().into_vec());
        assert_eq!(&got[..2], &[2, 3]);
    }

    /// A whole mutable HBM view uses the reshaped axes for subsequent writes.
    #[test]
    fn writes_reshaped_cells() {
        axes![X = 8, A = 4, B = 2];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut tensor = HbmTensor::<i32, m![1], m![X], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..8));
        // SAFETY: eight consecutive cells become four consecutive pairs, with no data movement.
        let view = unsafe { tensor.view_mut().reshape::<m![1], m![A, B]>() };
        view.tile::<m![A], 1, m![A = 1 #{!} 4, B]>(1)
            .write(&device.tdma, |mut dst| {
                dst.transpose(Tensor::<i32, m![B], Cpu>::from_vec([20, 30]).view(), false);
            });
        assert_eq!(tensor.load().into_vec(), [0, 1, 20, 30, 4, 5, 6, 7]);
    }

    /// An alias covers the selected cells after repeated whole-view reshapes.
    #[test]
    fn aliases_reshaped_cells() {
        axes![X = 8, A = 4, B = 2, C = 2];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let tensor = HbmTensor::<i32, m![1], m![X], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..8));
        // SAFETY: both reshapes preserve the order of the eight cells.
        let view = unsafe { tensor.view().reshape::<m![1], m![A, B]>().reshape::<m![1], m![A, C]>() };
        let tile = view.tile::<m![A], 1, m![A = 1 # 4, C]>(1);
        let start = tile.window().start;
        let buffer = crate::runtime::HbmBuffer(
            tile.inner
                .inner
                .buffer
                .alias(start..start + <m![C]>::SIZE * size_of::<i32>())
                .unwrap(),
        );
        let alias = HbmTensor::<i32, m![1], m![C], Cpu>::try_from(buffer).unwrap();
        assert_eq!(alias.load().into_vec(), [2, 3]);
    }

    /// A partial reshape retains the selected pair after a whole-view reshape.
    #[test]
    fn reads_reshaped_tile() {
        axes![X = 8, A = 4, B = 2, C = 2];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let tensor = HbmTensor::<i32, m![1], m![X], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..8));
        // SAFETY: eight consecutive cells become four consecutive pairs.
        let view = unsafe { tensor.view().reshape::<m![1], m![A, B]>() };
        let tile = view.tile::<m![A], 1, m![A = 1 # 4, B]>(2);
        // SAFETY: only the selected pair's inner axis is renamed, preserving its wire order.
        let got =
            unsafe { tile.reshape::<m![1], m![A = 1 # 4, C]>() }.read(&device.tdma, |cells| cells.read().into_vec());
        assert_eq!(&got[..2], &[4, 5]);
    }

    /// A partial mutable reshape writes only its nonzero-offset pair.
    #[test]
    fn writes_reshaped_tile() {
        axes![A = 4, B = 2, C = 2];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut tensor = HbmTensor::<i32, m![1], m![A, B], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..8));
        let tile = tensor.view_mut().tile::<m![A], 1, m![A = 1 #{!} 4, B]>(2);
        // SAFETY: only the selected pair's inner axis is renamed, preserving its wire order.
        unsafe { tile.reshape::<m![1], m![A = 1 #{!} 4, C]>() }.write(&device.tdma, |mut dst| {
            dst.transpose(Tensor::<i32, m![C], Cpu>::from_vec([40, 50]).view(), false);
        });
        assert_eq!(tensor.load().into_vec(), [0, 1, 2, 3, 40, 50, 6, 7]);
    }

    /// Runs an SPM-index gather and returns its values with a host oracle.
    fn run_gather<B: Backend>() -> (Vec<i32>, Vec<i32>) {
        axes![W = 8, V = 2, K = 128];
        type Slice = m![K / 2];
        // Padding-only PE axis spanning the PEs those slices cover: one PE here, 4 on a full cluster.
        type DummyPe = m![1 # { Slice::SIZE / constraints::SLICES_PER_PE }];
        // Non-monotonic and includes both table boundaries, exposing ignored indices.
        const PERM: [i32; 8] = [3, 7, 1, 5, 0, 6, 2, 4];
        let row = |k: usize| PERM[k % W::SIZE];

        let table_buf: Vec<i32> = (0..W::SIZE as i32).flat_map(|r| [10 * r, 10 * r + 1]).collect();
        let idx_buf: Vec<i32> = (0..K::SIZE).map(row).collect();
        let expected: Vec<i32> = (0..K::SIZE).flat_map(|k| [10 * row(k), 10 * row(k) + 1]).collect();

        let index_dm = DmTensor::<i32, m![1], m![1], Slice, m![K % 2], B>::from_parts(Tensor::from_vec(idx_buf));
        let mut tdma = DmaContext::<{ Dma::Tensor }, B>::on(
            B::open(vec![crate::Placement {
                chips: vec![0],
                pes: 0..4,
            }])
            .unwrap(),
        );
        let table = HbmTensor::<i32, m![1], m![W, V], B>::from_value(&tdma, &Tensor::from_vec(table_buf));
        let index: SpmTensor<i32, m![1], m![1], DummyPe, m![K], B> = index_dm.to_spm(&mut tdma);

        let output: DmTensor<i32, m![1], m![1], Slice, m![K % 2, V], B> = table
            .gather::<m![W], m![V]>()
            .by_positions::<m![K]>(&index)
            .to_dm::<m![1], Slice, m![K % 2, V]>(&mut tdma);
        (output.inner.into_vec(), expected)
    }

    #[test]
    fn cpu_gather_roundtrip() {
        let (got, expected) = run_gather::<Cpu>();
        assert_eq!(got, expected);
    }

    /// A deferred gather reads aliased HBM indices when its DMA executes.
    #[test]
    fn gathers_current_indices() {
        axes![W = 2, V = 2, K = 128];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![W, V], Cpu>::from_value(&device.tdma, &Tensor::from_vec([0, 1, 10, 11]));
        let index = HbmTensor::<i32, m![1], m![K], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec(std::iter::repeat_n(0, K::SIZE)),
        );
        let mut alias =
            HbmTensor::<i32, m![1], m![K], Cpu>::try_from(crate::HbmBuffer(index.buffer.alias(..).unwrap())).unwrap();
        let plan = table.gather::<m![W], m![V]>().by_byte_offsets::<m![K]>(&index);
        alias.store(&device.tdma, &Tensor::from_vec(std::iter::repeat_n(8, K::SIZE)));

        let output = plan.to_dm::<m![1], m![K / 2], m![K % 2, V]>(&mut device.tdma);

        assert_eq!(output.inner.into_vec(), [10, 11].repeat(K::SIZE));
    }

    /// A deferred scatter reads aliased HBM indices when its DMA executes.
    #[test]
    fn scatters_current_indices() {
        axes![W = 256, V = 2, K = 128];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut table = HbmTensor::<i32, m![1], m![W, V], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec(std::iter::repeat_n(-1, W::SIZE * V::SIZE)),
        );
        let index = HbmTensor::<i32, m![1], m![K], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec(std::iter::repeat_n(0, K::SIZE)),
        );
        let mut alias =
            HbmTensor::<i32, m![1], m![K], Cpu>::try_from(crate::HbmBuffer(index.buffer.alias(..).unwrap())).unwrap();
        let updates = DmTensor::<i32, m![1], m![1], m![K / 2], m![K % 2, V], Cpu>::from_parts(Tensor::from_vec(0..256));
        let plan = table
            .view_mut()
            .scatter::<m![W], m![V]>()
            .by_byte_offsets::<m![K]>(index.view());
        alias.store(&device.tdma, &Tensor::from_vec((1..=128).map(|row| row * 8)));

        plan.from_dm(&mut device.tdma, updates);

        let mut expected = vec![-1; W::SIZE * V::SIZE];
        expected[2..258].copy_from_slice(&(0..256).collect::<Vec<_>>());
        assert_eq!(table.load().into_vec(), expected);
    }

    /// A mutable HBM view borrows the owner's storage rather than a detached buffer. Once the view
    /// is released, reading the whole owning tensor must therefore observe the view's writes.
    #[test]
    fn cpu_hbm_view_mut_updates_owning_tensor() {
        axes![A = 8];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();

        let expected: Vec<i32> = (0..A::SIZE as i32).map(|x| x * 3 + 1).collect();
        let source = HbmTensor::<i32, m![1], m![A], Cpu>::from_value(&device.tdma, &Tensor::from_vec(expected.clone()));
        let mut destination = HbmTensor::<i32, m![1], m![A], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec(std::iter::repeat_n(-1, A::SIZE)),
        );

        let destination_view = destination.view_mut();
        source.view().read(&device.tdma, |source| {
            destination_view.write(&device.tdma, |mut destination| destination.transpose(source, false))
        });

        assert_eq!(destination.load().into_vec(), expected);
    }

    /// A device function's write that panics leaves the memory whole, as a device does.
    #[test]
    fn cpu_hbm_view_mut_keeps_memory_when_a_write_panics() {
        axes![A = 8];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();

        let values: Vec<i32> = (0..A::SIZE as i32).collect();
        let mut tensor =
            HbmTensor::<i32, m![1], m![A], Cpu>::from_value(&device.tdma, &Tensor::from_vec(values.clone()));
        let view = tensor.view_mut();
        let write = std::panic::AssertUnwindSafe(|| view.write(&device.tdma, |_| panic!("the write fails")));

        assert!(std::panic::catch_unwind(write).is_err());
        assert_eq!(tensor.load().into_vec(), values);
    }

    #[test]
    fn tile_windows_end_at_allocation() {
        axes![A = 8, H = 4];

        let row_bytes = H::SIZE * std::mem::size_of::<i32>();
        let allocation_bytes = A::SIZE * row_bytes;

        // The window a `launch` argument hands the device: one row in, out to the allocation's end.
        for start in [0, A::SIZE / 2, A::SIZE - 1] {
            assert_eq!(
                hbm_window::<i32, m![1]>(A::SIZE * H::SIZE, start * H::SIZE),
                start * row_bytes..allocation_bytes
            );
        }
    }

    #[test]
    fn uses_chip_tail() {
        axes![A = 16, C2 = 2, C4 = 4, P = 32];
        type Tail = m![1 # 16];
        type TailMut = m![1 #{!} 16];
        type PackedTail = m![P = 2 # 32];

        let one_chip = HbmTensor::<i32, m![1], m![A], Cpu>::new();
        assert_eq!(one_chip.view().tile::<m![A], 1, Tail>(1).window().len(), 60);

        let two_chips = HbmTensor::<i32, m![C2], m![A], Cpu>::new();
        assert_eq!(two_chips.view().tile::<m![A], 1, Tail>(1).window().len(), 60);

        let mut two_chips = HbmTensor::<i32, m![C2], m![A], Cpu>::new();
        assert_eq!(two_chips.view_mut().tile::<m![A], 1, TailMut>(1).window().len(), 60);

        let four_chips = HbmTensor::<i32, m![C4], m![A], Cpu>::new();
        assert_eq!(four_chips.view().tile::<m![A], 1, Tail>(1).window().len(), 60);

        let packed = HbmTensor::<i4, m![C2], m![P], Cpu>::new();
        assert_eq!(packed.view().tile::<m![P], 2, PackedTail>(2).window().len(), 15);
    }

    #[test]
    fn rejects_unaligned_view() {
        axes![C2 = 2, P = 32];
        type PackedTail = m![1 # 32];

        let packed = HbmTensor::<i4, m![C2], m![P], Cpu>::new();
        assert!(std::panic::catch_unwind(|| packed.view().tile::<m![P], 1, PackedTail>(1).window()).is_err());
    }

    #[test]
    fn rejects_invalid_view_partitions() {
        assert!(std::panic::catch_unwind(|| hbm_window::<i32, m![0]>(64, 0)).is_err());
        assert!(std::panic::catch_unwind(|| hbm_window::<i32, m![2]>(63, 0)).is_err());
        assert!(std::panic::catch_unwind(|| hbm_window::<i32, m![2]>(0, 0)).is_err());
    }

    /// A view of a view accumulates one offset per tile: each `tile` adds its own start, so the second
    /// start is relative to the first. Checked by reading the elements the nested view selects, since
    /// which cells a view names is what the accumulation decides.
    #[test]
    fn hbm_nested_tiles_accumulate_one_offset_each() {
        axes![A = 8, H = 4];
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();

        // Element `a * H + h` holds that flat position, so one value names the cell a view reached.
        let values: Vec<i32> = (0..(A::SIZE * H::SIZE) as i32).collect();
        let table = HbmTensor::<i32, m![1], m![A, H], Cpu>::from_value(&device.tdma, &Tensor::from_vec(values));

        let row = table.view().tile::<m![A], 1, m![1 # 8, H]>(3);
        let half = row.tile::<m![H], 2, m![1 # 8, H = 2 # 4]>(2);

        // The tile states its out-of-window cells as padding, so the read spans the declared mapping
        // and the two live cells come first.
        let read = half.read(&device.tdma, |half| half.read()).into_vec();
        assert_eq!(
            read[..2],
            [3 * H::SIZE as i32 + 2, 3 * H::SIZE as i32 + 3],
            "row 3, then 2 columns in"
        );
    }

    #[test]
    fn cpu_redistribution_accepts_an_equivalent_output_factorization() {
        axes![X = 4, Tail = 2];

        let input = DmTensor::<i32, m![1], m![1 # 2], m![1 # 64], m![X / 2 % 2, X % 2, Tail], Cpu>::from_parts(
            Tensor::from_vec((0..2 * 64 * X::SIZE * Tail::SIZE).map(|value| value as i32)),
        );
        let expected = input.view().inner.read().into_vec();
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let output = input
            .view()
            .chip_shuffle([0])
            .to_dm::<m![X % 4, Tail]>(&mut device.tdma);

        assert_eq!(output.view().inner.read().into_vec(), expected);
    }

    #[test]
    fn cpu_to_dm_moves_a_chip_axis_into_the_element() {
        axes![A = 2, C = 2];

        let input = DmTensor::<i32, m![A], m![1], m![1 # 64], m![C], Cpu>::from_parts(Tensor::from_vec(
            (0..A::SIZE * 64 * C::SIZE).map(|value| value as i32),
        ));
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let output = input.to_dm::<m![2], m![1], m![1 # 64], m![A, C]>(&mut device.tdma);

        let output_chip_size = 64 * A::SIZE * C::SIZE;
        let mut expected = vec![0; 2 * output_chip_size];
        expected[..4].copy_from_slice(&[0, 1, 128, 129]);
        expected[output_chip_size..output_chip_size + 4].copy_from_slice(&[0, 1, 128, 129]);
        assert_eq!(output.view().inner.read().into_vec(), expected);
    }
}

/// A host-to-HBM transfer, as a [`Prepared`] holds it; see [`HostTensor::to_hbm`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ToHbm<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend> {
    host: &'t HostTensor<D, Element, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    _marker: PhantomData<(Chip, Element2)>,
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend>
    ToHbm<'t, D, Element, Chip, Element2, B>
{
    /// Writes the host tensor into `hbm`, relaid out on the host first unless it already lays the
    /// cells out as `hbm` does, since a backend moves bytes unchanged.
    async fn write(self, hbm: &mut HbmTensor<D, Chip, Element2, B>) -> Result<(), Error> {
        if same_layout::<Element, Pair<Chip, Element2>>() {
            return self
                .dma
                .to_hbm([(B::bytes(self.host.storage()), hbm.memory_mut())])
                .await;
        }
        let staged: Tensor<D, m![{ Chip }, { Element2 }], B> = self.host.inner().transpose(true);
        self.dma.to_hbm([(B::bytes(&staged.inner), hbm.memory_mut())]).await
    }
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend> Work
    for ToHbm<'t, D, Element, Chip, Element2, B>
{
    type Output = HbmTensor<D, Chip, Element2, B>;
    type Future = Issued<impl Future<Output = Result<Self::Output, Error>>>;

    fn submit(self) -> Submitted<Self::Future> {
        async move {
            let mut hbm = HbmTensor::alloc_in(self.dma.device(), crate::runtime::Kind::Dram)?;
            self.write(&mut hbm).await?;
            Ok(hbm)
        }
        .submit()
    }
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend>
    Prepared<ToHbm<'t, D, Element, Chip, Element2, B>>
{
    /// Writes into `hbm`'s existing device allocation instead of a fresh one.
    pub fn output(
        self,
        hbm: &'t mut HbmTensor<D, Chip, Element2, B>,
    ) -> Prepared<impl Future<Output = Result<(), Error>>> {
        Prepared::new(self.into_work().write(hbm))
    }
}

/// An HBM-to-host transfer, as a [`Prepared`] holds it; see [`HbmTensor::to_host`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ToHost<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend> {
    hbm: &'t HbmTensor<D, Chip, Element, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    _marker: PhantomData<Element2>,
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend>
    ToHost<'t, D, Chip, Element, Element2, B>
{
    /// Reads the HBM tensor into `host`, relaid out on the host after unless `host` lays the cells
    /// out as the HBM tensor does, since a backend moves bytes unchanged.
    async fn read(self, host: &mut HostTensor<D, Element2, B>) -> Result<(), Error> {
        let from = self.hbm.memory();
        if same_layout::<Element2, Pair<Chip, Element>>() {
            return from.to_host(self.dma).output(B::bytes_mut(host.storage_mut())).await;
        }
        let mut staged = Tensor::<D, m![{ Chip }, { Element }], B>::zeroed();
        from.to_host(self.dma).output(B::bytes_mut(&mut staged.inner)).await?;
        *host = staged.transpose(true).into();
        Ok(())
    }
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend> Work
    for ToHost<'t, D, Chip, Element, Element2, B>
{
    type Output = HostTensor<D, Element2, B>;
    type Future = Issued<impl Future<Output = Result<Self::Output, Error>>>;

    fn submit(self) -> Submitted<Self::Future> {
        async move {
            let mut host = HostTensor::from(Tensor::zeroed());
            self.read(&mut host).await?;
            Ok(host)
        }
        .submit()
    }
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend>
    Prepared<ToHost<'t, D, Chip, Element, Element2, B>>
{
    /// Reads into `host`'s own memory instead of fresh memory.
    pub fn output(self, host: &'t mut HostTensor<D, Element2, B>) -> Prepared<impl Future<Output = Result<(), Error>>> {
        Prepared::new(self.into_work().read(host))
    }
}

/// Whether `A` and `B` lay the same cells out in the same order, so their bytes agree.
fn same_layout<A: M, B: M>() -> bool {
    A::SIZE == B::SIZE && A::to_value().normalize() == B::to_value().normalize()
}
