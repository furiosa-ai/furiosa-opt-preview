//! Tensors held in an engine's register file.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::constraints;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::CurrentBackend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::tensor::view::{TensorView, TensorViewMut};

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
    /// Restates the operand under a different mapping over the same register, consuming `self`. Its
    /// use is naming the slices a broadcast operand already holds a copy in, so it can feed a stream
    /// partitioned by a named axis (the operand rule is
    /// [`IntoBranchedOperand`](crate::engine::vector::operand::IntoBranchedOperand)'s).
    ///
    /// The VRF peer of [`DmTensor::reshape`](crate::tensor::memory::DmTensor::reshape); both delegate to [`Tensor::reshape`] and relabel rather
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
