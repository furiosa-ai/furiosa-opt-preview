//! Tensors placed in HBM: device memory with a type.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_lower::{PadInput, config_pad};
use furiosa_opt_macro::primitive;

use super::assert_chip_shuffle_pattern;
use crate::Error;
use crate::backend::{Allocation, Backend, Cpu};
use crate::constraints;
use crate::context::*;
use crate::runtime::{CurrentBackend, HbmBufferSlice, HbmBufferSliceMut, Region};
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::view::{TensorView, TensorViewMut, View, assert_tile};

/// Tensor stored in HBM memory: device memory with a type.
///
/// `TryFrom<crate::HbmBuffer>` takes a buffer whose per-chip size, chip count and alignment fit the
/// type; [`Self::into_buffer`] gives it back without copying.
#[primitive(HbmTensor)]
pub struct HbmTensor<D: Scalar, Chip: M, Element: M, B: Backend = CurrentBackend> {
    /// The memory this tensor names and keeps live: one chip's share of the bytes on every chip.
    pub(super) buffer: B::Buffer,
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
}

// ANCHOR: dma_impl
impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Reshapes the tensor to a different mapping at the same HBM address, consuming `self`.
    /// The HBM analogue of [`DmTensor::reshape`](crate::tensor::memory::DmTensor::reshape); both delegate to [`Tensor::reshape`].
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
    pub(crate) fn read<const DMA: Dma, R>(
        &self,
        dma: &DmaContext<DMA, B>,
        read: impl FnOnce(TensorView<'_, D, Pair<Chip, Element>, B>) -> R,
    ) -> R {
        let base = dma.load::<D>(self.inner.inner.clone());
        read(self.inner.window.over::<D, B>(&base))
    }

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(HbmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> HbmTensorView<'l, D, Chip2, Element, B, SomeChips> {
        assert_tile::<Index, Chip, Chip2, LEN>(PaddingKind::Top);
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
        assert_tile::<Index, Element, Element2, LEN>(PaddingKind::Top);
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
    /// outermost padding factor: `m![L1]` as `m![L1 # 256]`. See [`DmTensorView::pad`](crate::tensor::memory::DmTensorView::pad); on an HBM
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
    pub(super) inner: View<'l, D, Pair<Chip, Element>, B, HbmBufferSliceMut<'l, B>>,
    pub(super) chips: PhantomData<Chips>,
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
        assert_tile::<Index, Chip, Chip2, LEN>(PaddingKind::Bottom);
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
        assert_tile::<Index, Element, Element2, LEN>(PaddingKind::Bottom);
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
}
