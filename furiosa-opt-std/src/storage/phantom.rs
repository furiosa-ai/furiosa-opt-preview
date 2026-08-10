use std::marker::PhantomData;

use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::backend::op_prep::broadcast_axes;
use crate::scalar::*;

/// Typecheck tensor: metadata only (axes), no host buffer. All reads return `Opt::Uninit` and all
/// writes are silently dropped. Each op produces the (empty) phantom output of its declared layout,
/// with no per-element iteration.
///
/// Mapping legality is deliberately NOT re-derived here. Two reasons:
///
/// 1. The engine primitives already ran the shared `furiosa_opt_lower::config_*` verifiers, the same
///    single source of truth MIR→VISA translation calls (see `TuOperationBuilder::verify`), so
///    re-deriving here would duplicate a check the verifier owns.
/// 2. [`op_prep`]'s carve-based helpers are the VALUE backends' loop-bound prep, not a verifier: a
///    carve asks "does `dst` contain `src`", which is the shape of `BufStorage`'s write loop. A TE
///    transpose that drops padding (`W # 16 / 8 ⊗ W # 16 % 8` streamed out as `W = 12`) is a legal
///    op that fails that question, so running the helpers here would reject kernels the verifiers
///    and the compiler both accept.
///
/// What a Typecheck run rejects is therefore NARROWER than what the compiler rejects, and the gap is
/// not yet closed. `reduce` keeps `broadcast_axes`, which compares symbol PRESENCE only: a
/// destination factor whose symbol is present but whose stride, modulo, or extent does not line up
/// passes here. The carves removed above were catching that class, so it is pinned by test rather
/// than left undocumented.
///
/// [`op_prep`]: crate::backend::op_prep
#[derive(Clone, Debug)]
pub struct PhantomStorage<D: Scalar> {
    axes: Vec<AxisTerm>,
    _phantom: PhantomData<D>,
}

impl<D: Scalar> PartialEq for PhantomStorage<D> {
    fn eq(&self, other: &Self) -> bool {
        self.axes == other.axes
    }
}
impl<D: Scalar> Eq for PhantomStorage<D> {}

impl<D: Scalar> PhantomStorage<D> {
    /// An empty phantom tensor of the given `Mapping`'s layout, Typecheck carries axes only, no
    /// values.
    fn new<Mapping: M>() -> Self {
        Self::new_from_axes(Mapping::to_value().axes())
    }

    /// An empty phantom tensor carrying `axes` directly. Lets elementwise ops preserve the source
    /// axes without naming a mapping type.
    fn new_from_axes(axes: Vec<AxisTerm>) -> Self {
        Self {
            axes,
            _phantom: PhantomData,
        }
    }

    /// Typecheck carries no values; map/zip produce an empty phantom of the SAME axes, no
    /// per-element iteration. Back [`crate::backend::Backend::map`] / [`crate::backend::Backend::zip_with`].
    pub(crate) fn map<D2: Scalar>(&self, _f: impl FnMut(D) -> D2) -> PhantomStorage<D2> {
        PhantomStorage::new_from_axes(self.axes.clone())
    }

    pub(crate) fn zip_with<D2: Scalar, D3: Scalar>(
        &self,
        _other: &PhantomStorage<D2>,
        _f: impl Fn(D, D2) -> D3,
    ) -> PhantomStorage<D3> {
        PhantomStorage::new_from_axes(self.axes.clone())
    }

    /// Ternary peer of [`Self::zip_with`]; phantom carries no values.
    pub(crate) fn zip3_with<D2: Scalar, D3: Scalar, D4: Scalar>(
        &self,
        _b: &PhantomStorage<D2>,
        _c: &PhantomStorage<D3>,
        _f: impl Fn(D, D2, D3) -> D4,
    ) -> PhantomStorage<D4> {
        PhantomStorage::new_from_axes(self.axes.clone())
    }

    /// Typecheck no-op: the phantom `self` already carries the destination layout, and there are no
    /// values to move. Backs [`crate::backend::Backend::transpose`].
    pub(crate) fn transpose<Src: M, Mapping: M>(
        &mut self,
        _src: &PhantomStorage<D>,
        _src_offset: &Index,
        _dst_offset: &Index,
        _allow_broadcast: bool,
    ) {
    }

    /// Backs [`crate::backend::Backend::reduce`].
    pub(crate) fn reduce<Src: M, Dst: M, R: Fn(D, D) -> D>(
        &self,
        _reduce_fn: R,
        _identity: D,
        allow_broadcast: bool,
    ) -> Self {
        // `allow_broadcast` is the caller's declared intent, so an unintended new axis is a mismatch
        // between two type parameters, not a property of the value backends' loop: it stays checked
        // here (see the type doc for what does not).
        let broadcast = broadcast_axes(&Src::to_value(), &Dst::to_value());
        assert!(
            allow_broadcast || broadcast.axes().is_empty(),
            "reduce: Dst adds axes absent from the source; pass allow_broadcast=true for reduce-then-broadcast"
        );
        Self::new::<Dst>()
    }

    /// Backs [`crate::backend::Backend::contraction`]. Typecheck carries no values, so it neither
    /// widens nor folds; `out` is a runtime value here (the deferred engine-level fold only has it as
    /// one), so it is taken directly rather than rebuilt from a static mapping type.
    pub(crate) fn contraction(&self, _rhs: &Self, out: &MappingValue) -> Self {
        Self::new_from_axes(out.axes())
    }

    /// Typecheck no-op: no indices to decode, no values to move. Backs
    /// [`crate::backend::Backend::scatter`].
    pub(crate) fn scatter<Src: M, Key: M, Dst: M, Idx: M>(
        &self,
        _dst: &mut PhantomStorage<D>,
        _index: &PhantomStorage<i32>,
        _scaled: bool,
    ) {
    }

    /// Typecheck no-op; see [`Self::scatter`]. Backs [`crate::backend::Backend::gather`].
    pub(crate) fn gather<Src: M, Dst: M, Idx: M>(
        &self,
        _dst: &mut PhantomStorage<D>,
        _index: &PhantomStorage<i32>,
        _scaled: bool,
    ) {
    }

    /// Backs [`crate::backend::Backend::reshape`].
    pub(crate) fn reshape<Mapping: M, Mapping2: M>(&self) -> Self {
        assert_eq!(Mapping::SIZE, Mapping2::SIZE);
        Self::new::<Mapping2>()
    }

    /// Typecheck has no values, the input buffer is dropped regardless of its length. (Length is
    /// still validated by [`crate::tensor::Tensor::from_vec`] in the wrapper, unlike the `Opt` path.)
    pub(crate) fn from_vec(mapping: &MappingValue, _data: impl IntoIterator<Item = D>) -> Self {
        Self::new_from_axes(mapping.axes())
    }

    /// Typecheck carries no values.
    pub(crate) fn into_vec(self, _mapping: &MappingValue) -> Vec<D> {
        Vec::new()
    }

    /// An empty phantom tensor of this layout.
    pub(crate) fn zeroed(mapping: &MappingValue) -> Self {
        Self::new_from_axes(mapping.axes())
    }
}
