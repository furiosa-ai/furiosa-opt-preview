//! Tensors placed in data memory.

use furiosa_mapping::*;
use furiosa_opt_lower::{PadInput, config_pad};
use furiosa_opt_macro::primitive;

use super::{assert_asymmetric_slice, assert_slice_indices};
use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::view::{TensorView, TensorViewMut, Tiled, assert_tile};

/// Tensor stored in data memory.
#[primitive(DmTensor)]
#[derive(Debug)]
pub struct DmTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend = CurrentBackend> {
    pub(super) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Element>>>, B>,
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
pub(super) enum SliceIndexing {
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

    /// Creates immutable views by splitting along a tile expression over Chip.
    #[primitive(DmTensorView::chip_tile)]
    pub fn chip_tile<Index: M, const LEN: usize, Chip2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip2, Cluster, Slice, Element, B> {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster, Slice, Slice>();
        assert_tile::<Index, Chip, Chip2, LEN>(PaddingKind::Top);
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
        assert_tile::<Index, Cluster, Cluster2, LEN>(PaddingKind::Top);
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
        assert_tile::<Index, Slice, Slice2, LEN>(PaddingKind::Top);
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorView { inner }
    }

    /// Creates immutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorView::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        &self,
        start: usize,
    ) -> DmTensorView<'l, D, Chip, Cluster, Slice, Element2, B> {
        assert_tile::<Index, Element, Element2, LEN>(PaddingKind::Top);
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
    pub(super) fn reborrow(&mut self) -> DmTensorViewMut<'_, D, Chip, Cluster, Slice, Element, B> {
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
        assert_tile::<Index, Chip, Chip2, LEN>(PaddingKind::Bottom);
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
        assert_tile::<Index, Cluster, Cluster2, LEN>(PaddingKind::Bottom);
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
        assert_tile::<Index, Slice, Slice2, LEN>(PaddingKind::Bottom);
        let inner = self.inner.retile::<Index, _>(start, LEN);
        DmTensorViewMut { inner }
    }

    /// Creates mutable views by splitting along a tile expression over Element.
    #[primitive(DmTensorViewMut::tile)]
    pub fn tile<Index: M, const LEN: usize, Element2: M>(
        self,
        start: usize,
    ) -> DmTensorViewMut<'l, D, Chip, Cluster, Slice, Element2, B> {
        assert_tile::<Index, Element, Element2, LEN>(PaddingKind::Bottom);
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
