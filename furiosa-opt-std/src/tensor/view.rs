use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_lower::{PadInput, TileInput, TileMappingInput, config_pad, config_tile, tile_mapping};

use super::Tensor;
use crate::backend::{Backend, internal};
use crate::runtime::CurrentBackend;
use crate::scalar::*;

/// A tile mapping derived from its source mapping instead of supplied by an API caller.
#[derive(Debug, Clone)]
pub(crate) struct Tiled<Index, Element, const LEN: usize, const HOLE_FILL: PaddingKind>(PhantomData<(Index, Element)>);

impl<Index: M, Element: M, const LEN: usize, const HOLE_FILL: PaddingKind> M for Tiled<Index, Element, LEN, HOLE_FILL> {
    const SIZE: usize = Element::SIZE;

    fn to_value() -> Mapping {
        tile_mapping(TileMappingInput {
            index: Index::to_value(),
            element: Element::to_value(),
            len: LEN,
            hole_fill: HOLE_FILL,
        })
        .unwrap_or_else(|error| panic!("{error}"))
    }

    fn map(index: usize) -> Cell {
        Self::to_value().index(index)
    }
}

/// Panics if `E2`'s type-level derivation (e.g. a [`Tiled`] built from an invalid split) is not
/// well-formed. Does not, and cannot, check that `E2` actually corresponds to any particular
/// source mapping -- only that its own derivation holds together.
fn assert_valid_derivation<E2: M>() {
    E2::to_value();
}

/// Which cells of a base tensor a view covers, without the base's cells: an offset into the base,
/// with the view's own mapping `E` at the type level. A [`View`] pairs it with the cells; an HBM
/// view applies it to the bytes it loads from device memory with [`Self::over`].
#[derive(Debug, Clone)]
pub(crate) struct Window<E: M> {
    offset: Index,
    // The base tensor's live-axis mapping, set at construction and left untouched by `retile` (which
    // only narrows `E`). A tile records its axis in `offset` as a live `Symbol` but represents it in
    // `E` as padding; `base_map` keeps the axis live so a partial-view relayout can resolve the
    // offset's physical wire base against the real layout (see [`Backend::transpose`]).
    base_map: Mapping,
    _marker: PhantomData<E>,
}

impl<E: M> Window<E> {
    /// Every cell of a base tensor laid out as `E`.
    pub(crate) fn new() -> Self {
        Self {
            offset: Index::new(),
            base_map: E::to_value(),
            _marker: PhantomData,
        }
    }

    /// The window with `I` fixed at `start`, declared as `E2`; a start on padding panics, and `kind`
    /// says which padding a tile leaves around itself. `E2` is trusted, as
    /// [`TensorViewMut::retile`] describes.
    pub(crate) fn retile<I: M, E2: M>(self, start: usize, kind: PaddingKind) -> Window<E2> {
        let mut offset = self.offset;
        offset
            .add_mapping::<I>(start, kind)
            .unwrap_or_else(|kind| panic!("tile start maps to {kind:?} padding"));
        Window {
            offset,
            base_map: self.base_map,
            _marker: PhantomData,
        }
    }

    /// The same cells declared as `E2`, which the caller states moves no cell.
    pub(crate) fn redeclare<E2: M>(self) -> Window<E2> {
        Window {
            offset: self.offset,
            base_map: self.base_map,
            _marker: PhantomData,
        }
    }

    /// A whole view adopts the new axes; a partial view retains its owner's coordinates.
    pub(crate) fn reshape<E2: M>(self) -> Window<E2> {
        let base_map = reshaped_base::<E, E2>(self.base_map, &self.offset);
        Window {
            offset: self.offset,
            base_map,
            _marker: PhantomData,
        }
    }

    /// Element position where this window starts in the base tensor's wire layout, 0 for an untiled
    /// view. Walks the base mapping, so it is computed on demand rather than carried.
    pub(crate) fn window_base(&self) -> usize {
        crate::storage::window_base(&self.base_map, &self.offset)
    }

    /// Element count of the base tensor.
    pub(crate) fn base_len(&self) -> usize {
        self.base_map.size()
    }

    /// This window's cells of `base`, the storage of the tensor the window was taken from.
    pub(crate) fn over<'b, D: Scalar, B: Backend>(&self, base: &'b B::Storage<D>) -> TensorView<'b, D, E, B> {
        TensorView {
            inner: base,
            window: self.clone(),
            marker: PhantomData,
        }
    }

    /// [`Self::over`], to write.
    pub(crate) fn over_mut<'b, D: Scalar, B: Backend>(
        &self,
        base: &'b mut B::Storage<D>,
    ) -> TensorViewMut<'b, D, E, B> {
        TensorViewMut {
            inner: base,
            window: self.clone(),
            marker: PhantomData,
        }
    }
}

/// The cells of a base tensor a view covers: `inner` is where the base tensor's cells are, and
/// `window` which of them the view covers, in the view's current logical layout `E` (changed by
/// tiling). `S` is the only thing the kinds of view differ in:
/// - [`TensorView`] / [`TensorViewMut`]: the base tensor's host storage, which the view reads and
///   writes directly.
/// - an HBM view's device memory, whose cells a device function reads only through the backend.
pub struct View<'l, D: Scalar, E: M, B: Backend, S> {
    pub(crate) inner: S,
    pub(crate) window: Window<E>,
    pub(crate) marker: PhantomData<fn() -> (&'l (), D, B)>,
}

/// Mutable view into a tensor's host storage. Borrows the concrete storage of some base tensor
/// (mapping-agnostic after the storage-layer mapping erasure, so retiling is a plain rewrap of the
/// same borrow, no cast).
pub type TensorViewMut<'l, D, E, B = CurrentBackend> = View<'l, D, E, B, &'l mut <B as internal::Backend>::Storage<D>>;

impl<'l, D: Scalar, E: M, B: Backend, S: std::fmt::Debug> std::fmt::Debug for View<'l, D, E, B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("View")
            .field("inner", &self.inner)
            .field("offset", &self.window.offset)
            .finish()
    }
}

/// Immutable view into a tensor's host storage; see [`TensorViewMut`].
pub type TensorView<'l, D, E, B = CurrentBackend> = View<'l, D, E, B, &'l <B as internal::Backend>::Storage<D>>;

impl<'l, D: Scalar, E: M, B: Backend, S> View<'l, D, E, B, S> {
    /// Every cell of the base tensor `inner` holds.
    pub(crate) fn whole(inner: S) -> Self {
        Self {
            inner,
            window: Window::new(),
            marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, E: M, B: Backend, S: Clone> Clone for View<'l, D, E, B, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            window: self.window.clone(),
            marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, E: M, B: Backend> From<TensorViewMut<'l, D, E, B>> for TensorView<'l, D, E, B> {
    fn from(view: TensorViewMut<'l, D, E, B>) -> Self {
        Self {
            inner: view.inner,
            window: view.window,
            marker: PhantomData,
        }
    }
}

/// A whole view names the same buffer in the new axes, so subsequent tiles must resolve there.
/// A partial view keeps the owner's coordinates for the offset it already carries.
fn reshaped_base<E: M, E2: M>(base_map: Mapping, offset: &Index) -> Mapping {
    if offset.0.is_empty() && base_map == E::to_value() {
        E2::to_value()
    } else {
        base_map
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorViewMut<'l, D, E, B> {
    pub(crate) fn reborrow(&mut self) -> TensorViewMut<'_, D, E, B> {
        TensorViewMut {
            inner: &mut *self.inner,
            window: self.window.clone(),
            marker: PhantomData,
        }
    }

    /// Creates a new tensor view mut.
    pub(crate) fn new(inner: &'l mut B::Storage<D>) -> Self {
        Self {
            inner,
            window: Window::new(),
            marker: PhantomData,
        }
    }

    /// Splits the tensor view by tiling. As a write destination, the cells outside
    /// the tile must be down padding ([`PaddingKind::Bottom`]) so the commit
    /// sequencer never writes them.
    pub fn tile<I: M, E2: M, const LEN: usize>(self, start: usize) -> TensorViewMut<'l, D, E2, B> {
        config_tile(TileInput {
            index: I::to_value(),
            element: E::to_value(),
            expected: E2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Bottom,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        self.retile::<I, E2>(start)
    }

    /// Creates a mutable tile whose mapping is derived from the source mapping.
    pub(crate) fn tile_derived<I: M, const LEN: usize>(
        self,
        start: usize,
    ) -> TensorViewMut<'l, D, Tiled<I, E, LEN, { PaddingKind::Bottom }>, B> {
        self.retile_derived::<I, Tiled<I, E, LEN, { PaddingKind::Bottom }>>(start)
    }

    /// Rewraps a derived tile, panicking if its type-level split is invalid. `assert_valid_derivation`
    /// only catches a broken *derivation* (e.g. an invalid [`Tiled`]); it does not, and cannot, check
    /// that `E2` actually corresponds to `E` -- see [`Self::retile`]'s warning. Only call this with an
    /// `E2` derived from `E` by construction.
    pub(crate) fn retile_derived<I: M, E2: M>(self, start: usize) -> TensorViewMut<'l, D, E2, B> {
        assert_valid_derivation::<E2>();
        self.retile::<I, E2>(start)
    }

    /// The rewrap half of [`Self::tile`], for a caller that has already checked the split. A
    /// `DmTensorView` checks the one axis class its tile retargets, which its combined `E` cannot state.
    ///
    /// `E2` is trusted, not verified: only `I`'s position is checked against `E` (via the offset), and
    /// `E2` itself is recorded solely at the type level, with no comparison to `E` or the base mapping.
    /// Passing an `E2` that is not actually `E` with `I` fixed at `start` (byte-for-byte, including
    /// every other axis's stride) compiles and does not panic; it silently reads or writes the wrong
    /// addresses. Only pass an `E2` derived from the caller's *own* `E` by construction (see
    /// [`Tiled`], or a type checked the way [`Self::tile`] checks `E2` with [`config_tile`]) -- never a
    /// type merely declared to have the right shape.
    pub(crate) fn retile<I: M, E2: M>(self, start: usize) -> TensorViewMut<'l, D, E2, B> {
        TensorViewMut {
            inner: self.inner,
            window: self.window.retile::<I, E2>(start, PaddingKind::Bottom),
            marker: PhantomData,
        }
    }

    /// Reshapes the view to a different mapping `E2` over the same borrow, consuming `self`. A reshape
    /// is a MOVE: a rewrap of the same mapping-agnostic storage borrow (no data copied, no second
    /// handle). A whole view adopts the new axes as its base mapping for subsequent tiles. A partial
    /// view retains its existing offset and the owner's coordinate system.
    /// Sound on every backend, whose storage is mapping-agnostic (bare buffer / codegen relabel).
    ///
    /// # Safety
    ///
    /// `E::SIZE == E2::SIZE` is asserted below; the genuine precondition is that `E` and `E2` lay the
    /// elements out in the SAME physical (wire) order, so the relabel moves no data. Axis regrouping
    /// (merge/split) preserves wire order and is valid; a permutation is not (use [`Self::transpose`]).
    /// On a tiled view the relabel must preserve the live axes needed to resolve the existing
    /// offset and any later tiles against the owner's mapping.
    ///
    /// Kept a *runtime* `assert_eq!`, not a `const` block: every
    /// `{Dm,Hbm}TensorView(Mut)::reshape` delegates its `Element`-inclusive combined mapping here (see
    /// [`Tensor::reshape`]'s matching note -- a `const` block trips at monomorphization regardless of
    /// whether the call is ever reached at runtime).
    pub unsafe fn reshape<E2: M>(self) -> TensorViewMut<'l, D, E2, B> {
        assert_eq!(E::SIZE, E2::SIZE);
        TensorViewMut {
            inner: self.inner,
            window: self.window.reshape(),
            marker: PhantomData,
        }
    }

    /// Reads the mutable view without consuming it.
    pub(crate) fn read(&self) -> Tensor<D, E, B> {
        self.window.over::<D, B>(&*self.inner).read()
    }

    /// Transposes from a tensor. Delegates to [`Backend::transpose`], passing both storages and their
    /// live-axis maps so a partial-view relayout can resolve each offset's wire base.
    pub fn transpose<'lsrc, Src: M>(&mut self, src: TensorView<'lsrc, D, Src, B>, allow_broadcast: bool) {
        self.relayout(src, allow_broadcast, false);
    }

    pub(crate) fn relayout<Src: M>(
        &mut self,
        src: TensorView<'_, D, Src, B>,
        allow_broadcast: bool,
        preserve_copies: bool,
    ) {
        B::transpose::<D, Src, E>(
            self.inner,
            src.inner,
            &src.window.offset,
            &self.window.offset,
            &src.window.base_map,
            &self.window.base_map,
            allow_broadcast,
            preserve_copies,
        );
    }
}

impl<'l, D: Scalar, E: M, B: Backend> TensorView<'l, D, E, B> {
    /// Splits the tensor view by tiling. As a read source, the cells outside the
    /// tile stay accessible ([`PaddingKind::Top`]).
    pub fn tile<I: M, E2: M, const LEN: usize>(&self, start: usize) -> TensorView<'l, D, E2, B> {
        config_tile(TileInput {
            index: I::to_value(),
            element: E::to_value(),
            expected: E2::to_value(),
            len: LEN,
            hole_fill: PaddingKind::Top,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        self.retile::<I, E2>(start)
    }

    /// Creates a tile whose mapping is derived from the source mapping.
    pub(crate) fn tile_derived<I: M, const LEN: usize>(
        &self,
        start: usize,
    ) -> TensorView<'l, D, Tiled<I, E, LEN, { PaddingKind::Top }>, B> {
        self.retile_derived::<I, Tiled<I, E, LEN, { PaddingKind::Top }>>(start)
    }

    /// Rewraps a derived tile, panicking if its type-level split is invalid. `assert_valid_derivation`
    /// only catches a broken *derivation* (e.g. an invalid [`Tiled`]); it does not, and cannot, check
    /// that `E2` actually corresponds to `E` -- see [`Self::retile`]'s warning. Only call this with an
    /// `E2` derived from `E` by construction.
    pub(crate) fn retile_derived<I: M, E2: M>(&self, start: usize) -> TensorView<'l, D, E2, B> {
        assert_valid_derivation::<E2>();
        self.retile::<I, E2>(start)
    }

    /// The rewrap half of [`Self::tile`], for a caller that has already checked the split. See
    /// [`TensorViewMut::retile`] -- same trust, same silent-misaddress risk if `E2` is not truly `E`
    /// with `I` fixed at `start`.
    pub(crate) fn retile<I: M, E2: M>(&self, start: usize) -> TensorView<'l, D, E2, B> {
        TensorView {
            inner: self.inner,
            window: self.window.clone().retile::<I, E2>(start, PaddingKind::Top),
            marker: PhantomData,
        }
    }

    /// Creates a view at one position of each independent axis.
    pub(crate) fn retile_axes<E2: M>(&self, axes: &[Mapping], starts: &[usize]) -> TensorView<'l, D, E2, B> {
        assert_eq!(axes.len(), starts.len());
        let mut window = self.window.clone().redeclare::<E2>();
        for (axis, &start) in axes.iter().zip(starts) {
            match axis.index(start) {
                Cell::Index(index) => window.offset.add(index),
                Cell::Padding(kind) => panic!("tile start maps to {kind:?} padding"),
                Cell::OutOfBounds => panic!("tile start maps out of bounds"),
            }
        }
        TensorView {
            inner: self.inner,
            window,
            marker: PhantomData,
        }
    }

    /// Reshapes the view to a different mapping `E2` over the same borrow, consuming `self`. A reshape
    /// is a MOVE: a rewrap of the same mapping-agnostic storage borrow (no data copied, no second
    /// handle). A whole view adopts the new axes as its base mapping for subsequent tiles. A partial
    /// view retains its existing offset and the owner's coordinate system.
    /// Sound on every backend, whose storage is mapping-agnostic (bare buffer / codegen relabel).
    ///
    /// # Safety
    ///
    /// `E::SIZE == E2::SIZE` is asserted below; the genuine precondition is that `E` and `E2` lay the
    /// elements out in the SAME physical (wire) order, so the relabel moves no data. Axis regrouping
    /// (merge/split) preserves wire order and is valid; a permutation is not (use `transpose`).
    /// On a tiled view the relabel must preserve the live axes needed to resolve the existing
    /// offset and any later tiles against the owner's mapping.
    ///
    /// Kept a *runtime* `assert_eq!`, not a `const` block: every
    /// `{Dm,Hbm}TensorView(Mut)::reshape` delegates its `Element`-inclusive combined mapping here (see
    /// [`Tensor::reshape`]'s matching note -- a `const` block trips at monomorphization regardless of
    /// whether the call is ever reached at runtime).
    pub unsafe fn reshape<E2: M>(self) -> TensorView<'l, D, E2, B> {
        assert_eq!(E::SIZE, E2::SIZE);
        TensorView {
            inner: self.inner,
            window: self.window.reshape(),
            marker: PhantomData,
        }
    }

    /// Views the same cells inside the wider buffer they sit in: `m![A, B]` as `m![A # n, B]`. Safe for
    /// the reason a `Top` pad is: the added cells are declared don't-care, so a read never resolves one.
    ///
    /// Padding only at the OUTERMOST position ([`config_pad`]), which leaves every cell where it is.
    /// Use it where a consumer needs the extent stated and the view's type has dropped it, such as the
    /// two operands of an interleave, which must agree on one `Element`.
    pub fn pad<E2: M>(self) -> TensorView<'l, D, E2, B> {
        config_pad(PadInput {
            element: E::to_value(),
            expected: E2::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        self.redeclare()
    }

    /// Views a padded buffer's live cells alone: `m![A # n, B]` as `m![A, B]`, the inverse of
    /// [`Self::pad`]. Address-preserving for the same reason: an OUTERMOST pad sits above every live
    /// cell, so dropping it moves none of them and only stops the declaration short of the pad.
    ///
    /// This is what a consumer of a wider-staged producer needs, and it is NOT the same as tiling the
    /// live rows out: a tile leaves the base buffer as the source, so a fetch from it still spans the
    /// extent. An unpad re-declares the buffer, so the fetch spans the live cells alone.
    pub fn unpad<E2: M>(self) -> TensorView<'l, D, E2, B> {
        config_pad(PadInput {
            element: E2::to_value(),
            expected: E::to_value(),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        self.redeclare()
    }

    /// The rewrap [`Self::pad`] and [`Self::unpad`] share, with no check of its own. Callers that hold
    /// the mapping CLASSES apart (`DmTensorView`) check the one class whose extent changes and come here
    /// for the rewrap, since padding their combined mapping at the outermost would pad the distribution
    /// axes instead of the buffer.
    pub(crate) fn redeclare<E2: M>(self) -> TensorView<'l, D, E2, B> {
        TensorView {
            inner: self.inner,
            window: self.window.redeclare(),
            marker: PhantomData,
        }
    }

    /// Reads the tensor view into a new tensor. Delegates to [`Backend::transpose`], so the read is a
    pub fn read(self) -> Tensor<D, E, B> {
        let mut result = Tensor::zeroed();
        result.view_mut().transpose(self, false);
        result
    }

    /// Creates a new tensor view.
    pub(crate) fn new(inner: &'l B::Storage<D>) -> Self {
        Self {
            inner,
            window: Window::new(),
            marker: PhantomData,
        }
    }
}
