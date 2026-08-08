//! Operands for VE binary and ternary ops: [`VeRhs`] (constant, VRF or stash), [`Stash`] itself,
//! [`BranchOperands`] for a per-branch list, the [`BinaryOperandTag`] / [`TernaryOperandTag`] pair
//! and the [`IntoOperands`] conversions that let a call site pass a bare value.
//!
//! [`StashTransition`] is how the operand moves the pipeline's stash typestate.

use std::marker::PhantomData;

use furiosa_mapping::{M, Pair, m};
use furiosa_opt_macro::primitive;

use crate::engine::vector::stage::markers::Way;
use crate::{
    array_vec::ArrayVec,
    engine::vector::{
        MAX_TAGS,
        scalar::VeScalar,
        stage::state::VeState,
        stash_slot::{Occupied, Spent, StashState},
    },
    prelude::{GroupId, TagFilter, VrfTensor},
    tensor::Tensor,
};

// ============================================================================
// VeRhs - Constant or VRF operand (type-safe)
// ============================================================================

/// RHS operand for Vector Engine operations. `TargetMapping` is the shape a VRF operand is
/// transposed to.
#[primitive(op::VeRhs)]
#[derive(Debug, Clone)]
pub enum VeRhs<D: VeScalar, TargetMapping: M> {
    /// Constant value.
    Const {
        /// The constant value.
        v: D,
    },
    /// VRF data that has been transposed to match the target tensor shape.
    Vrf {
        /// The transposed VRF tensor.
        data: Tensor<D, TargetMapping>,
    },
    /// Read from stash (previously written value).
    Stash,
}

impl<D: VeScalar, TargetMapping: M> VeRhs<D, TargetMapping> {
    /// Creates a constant operand.
    #[primitive(op::VeRhs::constant)]
    pub fn constant(v: D) -> Self {
        VeRhs::Const { v }
    }

    /// Creates a VeRhs from a VrfTensor, automatically transposing to match the target tensor shape.
    #[primitive(op::VeRhs::vrf)]
    pub fn vrf<Chip: M, Cluster: M, Slice: M, Element: M>(vrf: &VrfTensor<D, Chip, Cluster, Slice, Element>) -> Self {
        let transposed = vrf.inner.transpose::<TargetMapping>(true);
        VeRhs::Vrf { data: transposed }
    }
}

impl<TargetMapping: M> From<i32> for VeRhs<i32, TargetMapping> {
    fn from(v: i32) -> Self {
        VeRhs::Const { v }
    }
}

impl<TargetMapping: M> From<f32> for VeRhs<f32, TargetMapping> {
    fn from(v: f32) -> Self {
        VeRhs::Const { v }
    }
}

impl<D: VeScalar, TargetMapping: M> From<Stash> for VeRhs<D, TargetMapping> {
    fn from(_: Stash) -> Self {
        VeRhs::Stash
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, TargetMapping: M>
    From<&VrfTensor<D, Chip, Cluster, Slice, Element>> for VeRhs<D, TargetMapping>
{
    fn from(vrf: &VrfTensor<D, Chip, Cluster, Slice, Element>) -> Self {
        VeRhs::vrf(vrf)
    }
}

/// The operand that reads the stash: `vector_fp_binary(op, Stash)`, `vector_fp_ternary(op,
/// (Stash, c))`. A value read more than once is not a stash -- use a read-many [`VrfTensor`].
#[primitive(op::Stash)]
#[derive(Debug, Clone, Copy)]
pub struct Stash;

mod sealed {
    /// Private supertrait of [`Plain`](super::Plain): a new operand type cannot silently be treated
    /// as stash-agnostic. Adding an operand means an explicit `impl Plain` (identity stash
    /// transition) or an explicit [`StashTransition`](super::StashTransition), never a default.
    pub trait Sealed {}
}

/// An operand that leaves the stash slot alone (everything but [`Stash`]). Rust has no negative
/// bounds, so the identity [`StashTransition`] blanket needs this marker to keep off `Stash`. Sealed,
/// so a new operand type cannot take the identity by forgetting to say anything.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be used as an operand here",
    label = "not usable as this op's operand",
    note = "if this is `Stash`: a `vector_stash` has to come earlier, only one op may read it, and the \
            stream has to carry the same scalar and way it carried at that write (see `StashTransition`)"
)]
pub trait Plain: sealed::Sealed {}

/// How an operand moves the stash slot: `S` -> [`Self::Next`], for a reader of scalar `RD` on a
/// way-`W` pipeline. [`Plain`] operands are the identity; [`Stash`] is impl'd only on
/// `Occupied<RD, _, W>`, so an empty, spent, repeated, cross-scalar or cross-way read has no impl.
/// See the stash-slot [module docs] for the state machine.
///
/// [module docs]: crate::engine::vector::stash_slot
///
/// A live stash at the reader's scalar and way reads:
/// ```
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Occupied, Stash, StashState, StashTransition, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Occupied<f32, Broadcast<1>, { Way::Way8 }>, f32, { Way::Way8 }, Stash>();
/// ```
/// A stash written as `i32` cannot be read by an `f32` op, so undo the conversion or reinterpret first:
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Occupied, Stash, StashState, StashTransition, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Occupied<i32, Broadcast<1>, { Way::Way8 }>, f32, { Way::Way8 }, Stash>();
/// ```
/// Nor 4-way by an 8-way op: the operand register is addressed per way, so the reader would get the
/// wrong elements.
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Occupied, Stash, StashState, StashTransition, VeScalar};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Occupied<f32, Broadcast<1>, { Way::Way4 }>, f32, { Way::Way8 }, Stash>();
/// ```
/// Never written, and read twice, are the same missing impl:
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Fresh, Stash, StashState, StashTransition, VeScalar};
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Fresh, f32, { Way::Way8 }, Stash>();
/// ```
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, Spent, Stash, StashState, StashTransition, VeScalar};
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Spent, f32, { Way::Way8 }, Stash>();
/// ```
#[diagnostic::on_unimplemented(
    message = "the stash cannot be read here",
    label = "this operand reads the stash",
    note = "the stash keeps the scalar and the way the stream carried at the `vector_stash`, and only a \
            stream carrying that same pair can read it: reinterpret (or narrow / widen) back first",
    note = "it is written once and read once, so with no earlier `vector_stash` the slot is empty, and a \
            second `Stash` operand finds it spent"
)]
pub trait StashTransition<S: StashState, RD: VeScalar, const W: Way> {
    /// Stash typestate after this operand.
    type Next: StashState;

    /// Runs the transition on the value (drops the stash tensor iff this operand read it). Needed
    /// because the state stores the stash by value, so `S` -> `Next` is a real move, not a retag.
    fn transition(state: VeState<S>) -> VeState<Self::Next>;

    /// The stashed tensor this operand reads, in the reader's mapping, or `None` for an operand that
    /// does not read the stash. Which one it is follows from the operand type, so the emulation needs
    /// no runtime look at the lowered operands.
    fn stashed<Mapping: M>(state: &VeState<S>) -> Option<Tensor<RD, Mapping>>;
}

impl<S: StashState, RD: VeScalar, T: Plain, const W: Way> StashTransition<S, RD, W> for T {
    type Next = S;
    fn transition(state: VeState<S>) -> VeState<S> {
        state
    }
    fn stashed<Mapping: M>(_state: &VeState<S>) -> Option<Tensor<RD, Mapping>> {
        None
    }
}

impl<D: VeScalar, StashMapping: M, const W: Way> StashTransition<Occupied<D, StashMapping, W>, D, W> for Stash {
    type Next = Spent;
    fn transition(state: VeState<Occupied<D, StashMapping, W>>) -> VeState<Spent> {
        state.consume_stash()
    }
    fn stashed<Mapping: M>(state: &VeState<Occupied<D, StashMapping, W>>) -> Option<Tensor<D, Mapping>> {
        Some(state.stash_tensor())
    }
}

// Ternary `(Stash, c)`: same read-once rule; the stash is operand0. Delegates to the `Stash`
// transition so the two share one body.
impl<D: VeScalar, StashMapping: M, const W: Way> StashTransition<Occupied<D, StashMapping, W>, D, W> for (Stash, f32) {
    type Next = Spent;
    fn transition(state: VeState<Occupied<D, StashMapping, W>>) -> VeState<Spent> {
        <Stash as StashTransition<Occupied<D, StashMapping, W>, D, W>>::transition(state)
    }
    fn stashed<Mapping: M>(state: &VeState<Occupied<D, StashMapping, W>>) -> Option<Tensor<D, Mapping>> {
        <Stash as StashTransition<Occupied<D, StashMapping, W>, D, W>>::stashed(state)
    }
}

// ============================================================================
// Multi-branch operands with a per-branch stash read
// ============================================================================

/// Type-level marker: this multi-branch operand list contains **no** branch that reads the stash,
/// so it leaves the slot alone. See [`BranchOperands`].
#[derive(Debug, Clone, Copy)]
pub struct NoStash;

/// Marks a [`BranchOperands`] list with at least one stash branch. The branches are mutually
/// exclusive, so any number of them collapse to one `Occupied` -> `Spent`.
#[derive(Debug, Clone, Copy)]
pub struct WithStash;

/// A multi-branch VE operand list where each branch picks its own rhs, and any branch may read the
/// stash.
///
/// One VE op is one pass: every element takes the branch whose [`TagFilter`] matches its execution
/// id, and the branches partition the elements. So several branches naming [`Stash`] still read the
/// hardware stash once, which is why read-once holds per pass rather than per branch. `StashMark`
/// records which case this list is ([`WithStash`] after a [`stash_branch`](Self::stash_branch),
/// [`NoStash`] otherwise) and the [`StashTransition`] keys on it.
///
/// Two branches reading the stash in one op is one read:
/// ```
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, BranchOperands, GroupId, Occupied, StashState, StashTransition, VeScalar, WithStash};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Occupied<f32, Broadcast<1>, { Way::Way8 }>, f32, { Way::Way8 }, BranchOperands<f32, Broadcast<1>, WithStash>>();
/// let _ = BranchOperands::<f32, Broadcast<1>, _>::new().stash_branch(GroupId::Zero).stash_branch(GroupId::One);
/// ```
/// Reading it again in a later op is not, and neither is reading it before any write:
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, BranchOperands, Spent, StashState, StashTransition, VeScalar, WithStash};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Spent, f32, { Way::Way8 }, BranchOperands<f32, Broadcast<1>, WithStash>>();
/// ```
/// ```compile_fail,E0277
/// # #![feature(adt_const_params)]
/// # use furiosa_opt_std::prelude::{stage::Way, BranchOperands, Fresh, StashState, StashTransition, VeScalar, WithStash};
/// # use furiosa_mapping::Broadcast;
/// # fn reads<S: StashState, RD: VeScalar, const W: Way, Op: StashTransition<S, RD, W>>() {}
/// reads::<Fresh, f32, { Way::Way8 }, BranchOperands<f32, Broadcast<1>, WithStash>>();
/// ```
#[derive(Debug, Clone)]
pub struct BranchOperands<D: VeScalar, Mapping: M, StashMark> {
    tags: ArrayVec<OperandTagValue<D, Mapping, ()>, MAX_TAGS>,
    _mark: PhantomData<StashMark>,
}

impl<D: VeScalar, Mapping: M> BranchOperands<D, Mapping, NoStash> {
    /// Starts an empty branch list. No branch reads the stash yet.
    pub fn new() -> Self {
        Self {
            tags: ArrayVec::empty(),
            _mark: PhantomData,
        }
    }
}

impl<D: VeScalar, Mapping: M> Default for BranchOperands<D, Mapping, NoStash> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: VeScalar, Mapping: M, StashMark> BranchOperands<D, Mapping, StashMark> {
    /// Pushes a non-stash branch (const or `&VrfTensor` rhs) gated on `group`. Leaves the stash
    /// marker unchanged.
    pub fn branch(mut self, rhs: impl Into<VeRhs<D, Mapping>>, group: GroupId) -> Self {
        self.tags.push(BinaryOperandTag::group(rhs.into(), group));
        self
    }

    /// Pushes a branch whose rhs reads the stash, gated on `group`. Flips the list to [`WithStash`]
    /// so the whole op's [`StashTransition`] consumes the slot once. A `NoStash` list becomes
    /// `WithStash`; a list that already reads the stash stays `WithStash` (the reads are mutually
    /// exclusive, so still one dynamic read).
    pub fn stash_branch(self, group: GroupId) -> BranchOperands<D, Mapping, WithStash> {
        let mut tags = self.tags;
        tags.push(BinaryOperandTag::group(VeRhs::Stash, group));
        BranchOperands {
            tags,
            _mark: PhantomData,
        }
    }
}

// A `NoStash` list touches nothing, so it is `Plain` (identity transition) like any other
// stash-free operand.
impl<D: VeScalar, Mapping: M> sealed::Sealed for BranchOperands<D, Mapping, NoStash> {}
impl<D: VeScalar, Mapping: M> Plain for BranchOperands<D, Mapping, NoStash> {}

// A `WithStash` list reads the stash once for the whole pass: `Occupied` -> `Spent`, impl'd only on
// `Occupied` so a second stash read anywhere later in the chain has no impl (read-once).
impl<D: VeScalar, StashMapping: M, Mapping: M, const W: Way> StashTransition<Occupied<D, StashMapping, W>, D, W>
    for BranchOperands<D, Mapping, WithStash>
{
    type Next = Spent;
    fn transition(state: VeState<Occupied<D, StashMapping, W>>) -> VeState<Spent> {
        state.consume_stash()
    }
    fn stashed<TargetMapping: M>(state: &VeState<Occupied<D, StashMapping, W>>) -> Option<Tensor<D, TargetMapping>> {
        <Stash as StashTransition<Occupied<D, StashMapping, W>, D, W>>::stashed(state)
    }
}

impl<D: VeScalar, TargetMapping: M, StashMark> IntoOperands<D, TargetMapping>
    for BranchOperands<D, TargetMapping, StashMark>
{
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS> {
        let always_count = self
            .tags
            .iter()
            .filter(|op| matches!(op.tag_filter(), TagFilter::All))
            .count();
        assert!(
            always_count <= 1,
            "Multiple All operands are not allowed (found {always_count})"
        );
        self.tags
    }
}

// Leaf operands that leave the stash slot alone.
impl sealed::Sealed for i32 {}
impl Plain for i32 {}
impl sealed::Sealed for f32 {}
impl Plain for f32 {}
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> sealed::Sealed
    for &VrfTensor<D, Chip, Cluster, Slice, Element>
{
}
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> Plain for &VrfTensor<D, Chip, Cluster, Slice, Element> {}
impl<D: VeScalar, Mapping: M> sealed::Sealed for BinaryOperandTag<D, Mapping> {}
impl<D: VeScalar, Mapping: M> Plain for BinaryOperandTag<D, Mapping> {}
impl<Mapping: M> sealed::Sealed for TernaryOperandTag<Mapping> {}
impl<Mapping: M> Plain for TernaryOperandTag<Mapping> {}

// Composites are [`Plain`] iff every element is: a stash read (`Stash`) is never `Plain`, so a
// tuple/array/`ArrayVec` containing one is not `Plain` and keeps its explicit `StashTransition`.
impl<T: Plain, const N: usize> sealed::Sealed for [T; N] {}
impl<T: Plain, const N: usize> Plain for [T; N] {}
impl<T: Plain, const N: usize> sealed::Sealed for ArrayVec<T, N> {}
impl<T: Plain, const N: usize> Plain for ArrayVec<T, N> {}
impl<A: Plain, B: Plain> sealed::Sealed for (A, B) {}
impl<A: Plain, B: Plain> Plain for (A, B) {}

// ============================================================================
// StashOperand - Stash read with branch validity (type-safe)
// ============================================================================

/// Stash operand for Vector Engine operations.
#[derive(Debug, Clone)]
pub struct StashOperand<D: VeScalar> {
    pub(crate) tag_filter: TagFilter,
    _phantom: PhantomData<D>,
}

impl<D: VeScalar> StashOperand<D> {
    pub(crate) fn always() -> Self {
        Self {
            tag_filter: TagFilter::All,
            _phantom: PhantomData,
        }
    }

    #[expect(dead_code)]
    pub(crate) fn group(id: GroupId) -> Self {
        Self {
            tag_filter: TagFilter::Group { id },
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// OperandTagValue - Operand carrying branch-gating predicate plus rhs values
// ============================================================================

/// Per-branch operand for VE binary / ternary ops: `x < 0 -> add(x, 1)`, `x >= 0 -> add(x, 2)`.
/// `operand0` is the primary rhs and applies to both shapes; `operand1` is the ternary's second one
/// (`()` in the binary alias). `tag_filter` picks the branch ids this operand applies at, and an
/// `Uninit` execution id skips it. See [`BinaryOperandTag`] / [`TernaryOperandTag`].
#[derive(Debug, Clone)]
pub struct OperandTagValue<D: VeScalar, TargetMapping: M, Operand1: Copy> {
    /// Primary rhs (always present). Replaces mainstream at this branch.
    pub operand0: VeRhs<D, TargetMapping>,
    /// Secondary rhs scalar. `()` for binary, `f32` for ternary.
    pub operand1: Operand1,
    /// Predicate gating which branch ids actually apply this operand.
    pub tag_filter: TagFilter,
}

/// Per-branch operand for binary VE operations: `operand0` only, `operand1 = ()`.
#[primitive(op::BinaryOperandTag)]
pub type BinaryOperandTag<D, TargetMapping> = OperandTagValue<D, TargetMapping, ()>;

/// Per-branch operand for ternary VE operations: `operand0` (rhs) plus `operand1: f32`.
/// Ternary ops are only supported for f32 tensors.
pub type TernaryOperandTag<Mapping> = OperandTagValue<f32, Mapping, f32>;

impl<D: VeScalar, TargetMapping: M> OperandTagValue<D, TargetMapping, ()> {
    /// Creates an always-valid operand.
    #[primitive(op::BinaryOperandTag::always)]
    pub fn always(operand0: VeRhs<D, TargetMapping>) -> Self {
        Self {
            operand0,
            operand1: (),
            tag_filter: TagFilter::All,
        }
    }

    /// Creates a group-specific operand.
    pub fn group(operand0: VeRhs<D, TargetMapping>, id: GroupId) -> Self {
        Self {
            operand0,
            operand1: (),
            tag_filter: TagFilter::Group { id },
        }
    }
}

// No public `stash_*` tag constructor: a tag is `Plain` (identity stash transition), so a
// stash-carrying tag would bypass the read-once typestate. The only stash-read path is the tracked
// `Stash` operand, which flows through `From<Stash> for VeRhs` and the `StashTransition` machinery.

// ============================================================================
// TernaryOperandTag - For ternary operations (f32 only)
// ============================================================================

impl<Mapping: M> OperandTagValue<f32, Mapping, f32> {
    /// Creates a TernaryOperandTag always valid.
    pub fn always(operand0: VeRhs<f32, Mapping>, operand1: f32) -> Self {
        Self {
            operand0,
            operand1,
            tag_filter: TagFilter::All,
        }
    }

    /// Creates a TernaryOperandTag valid for a specific group.
    pub fn group(operand0: VeRhs<f32, Mapping>, operand1: f32, id: GroupId) -> Self {
        Self {
            operand0,
            operand1,
            tag_filter: TagFilter::Group { id },
        }
    }
}

/// One view over [`BinaryOperandTag`] and [`TernaryOperandTag`] so the apply helpers iterate either
/// through one path. Only the per-branch rhs values are here; the mainstream is a tensor-level input
/// the helper takes separately. `Operand1` is `()` for the binary alias and `f32` for the ternary
/// one, so a caller that knows the shape gets a typed value rather than an `Option`.
pub trait OperandTag<D: VeScalar, Mapping: M> {
    /// Type of the secondary rhs scalar (`()` when the operand kind doesn't carry one).
    type Operand1: Copy;
    /// Primary rhs value (always present).
    fn operand0(&self) -> &VeRhs<D, Mapping>;
    /// Secondary rhs scalar.
    fn operand1(&self) -> Self::Operand1;
    /// Predicate gating whether this operand applies at a given execution id.
    fn tag_filter(&self) -> &TagFilter;
}

impl<D: VeScalar, Mapping: M, Operand1: Copy> OperandTag<D, Mapping> for OperandTagValue<D, Mapping, Operand1> {
    type Operand1 = Operand1;
    fn operand0(&self) -> &VeRhs<D, Mapping> {
        &self.operand0
    }
    fn operand1(&self) -> Operand1 {
        self.operand1
    }
    fn tag_filter(&self) -> &TagFilter {
        &self.tag_filter
    }
}

// From implementations for TernaryOperandTag (enables blanket impl for IntoGroupTernaryOperandTag)

/// `(Into<VeRhs<f32, Mapping>>, f32)` - VeRhs and constant become TernaryOperandTag.
impl<R, Mapping: M> From<(R, f32)> for TernaryOperandTag<Mapping>
where
    R: Into<VeRhs<f32, Mapping>>,
{
    fn from((operand0, operand1): (R, f32)) -> Self {
        TernaryOperandTag::always(operand0.into(), operand1)
    }
}

impl<R, B, Mapping: M> From<((R, f32), B)> for TernaryOperandTag<Mapping>
where
    R: Into<VeRhs<f32, Mapping>>,
    B: Into<TagFilter>,
{
    fn from(((operand0, operand1), branch): ((R, f32), B)) -> Self {
        TernaryOperandTag {
            operand0: operand0.into(),
            operand1,
            tag_filter: branch.into(),
        }
    }
}

// ============================================================================
// IntoTernaryOperandTags trait (for ternary operations, f32 only)
// ============================================================================

/// Lowers a ternary operand to `ArrayVec<TernaryOperandTag>`: anything `Into<TernaryOperandTag>`
/// (`(f32, f32)`, `(VeRhs, f32)`, `(Stash, f32)`, a single tag), or an array of tags per branch.
pub trait IntoTernaryOperandTags<TargetMapping: M> {
    /// Converts into an ArrayVec of TernaryOperandTag.
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS>;
}

// Blanket impl: Into<TernaryOperandTag> automatically provides IntoTernaryOperandTags
impl<T, TargetMapping: M> IntoTernaryOperandTags<TargetMapping> for T
where
    T: Into<TernaryOperandTag<TargetMapping>>,
{
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
        ArrayVec::new([self.into()])
    }
}

/// Array of `TernaryOperandTag` for multi-branch operations.
impl<TargetMapping: M, const N: usize> IntoTernaryOperandTags<TargetMapping> for [TernaryOperandTag<TargetMapping>; N] {
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
        // Validate: at most one All operand is allowed
        let always_count = self.iter().filter(|op| matches!(op.tag_filter, TagFilter::All)).count();
        assert!(
            always_count <= 1,
            "Multiple All operands are not allowed (found {always_count})"
        );
        ArrayVec::new(self)
    }
}

/// `ArrayVec<TernaryOperandTag, MAX_TAGS>` passes through.
impl<TargetMapping: M> IntoTernaryOperandTags<TargetMapping> for ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
    fn into_ternary_operands(self) -> ArrayVec<TernaryOperandTag<TargetMapping>, MAX_TAGS> {
        self
    }
}

// ============================================================================
// From implementations for BinaryOperandTag (enables .into() conversion)
// ============================================================================

impl<R, D: VeScalar, Mapping: M> From<R> for BinaryOperandTag<D, Mapping>
where
    R: Into<VeRhs<D, Mapping>>,
{
    fn from(rhs: R) -> Self {
        BinaryOperandTag::always(rhs.into())
    }
}

impl<R, B, D: VeScalar, Mapping: M> From<(R, B)> for BinaryOperandTag<D, Mapping>
where
    R: Into<VeRhs<D, Mapping>>,
    B: Into<TagFilter>,
{
    fn from((rhs, branch): (R, B)) -> Self {
        BinaryOperandTag {
            operand0: rhs.into(),
            operand1: (),
            tag_filter: branch.into(),
        }
    }
}

// ============================================================================
// IntoOperands trait - Multiple operands conversion
// ============================================================================

/// Lowers a binary operand to `ArrayVec<BinaryOperandTag>`. A single one (`i32`, `f32`,
/// `&VrfTensor`, [`Stash`], a tag) is wrapped; `[BinaryOperandTag; N]` and `ArrayVec` pass through
/// for the per-branch case.
pub trait IntoOperands<D: VeScalar, TargetMapping: M> {
    /// Converts into an ArrayVec of operands.
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS>;
}

// Blanket impl: Into<BinaryOperandTag> automatically provides IntoOperands
impl<T, D: VeScalar, TargetMapping: M> IntoOperands<D, TargetMapping> for T
where
    T: Into<BinaryOperandTag<D, TargetMapping>>,
{
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS> {
        ArrayVec::new([self.into()])
    }
}

impl<D: VeScalar, TargetMapping: M> IntoOperands<D, TargetMapping>
    for ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS>
{
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS> {
        self
    }
}

impl<D: VeScalar, TargetMapping: M, const N: usize> IntoOperands<D, TargetMapping>
    for [BinaryOperandTag<D, TargetMapping>; N]
{
    fn into_operands(self) -> ArrayVec<BinaryOperandTag<D, TargetMapping>, MAX_TAGS> {
        // Validate: at most one All operand is allowed
        let always_count = self
            .iter()
            .filter(|op| matches!(op.tag_filter(), TagFilter::All))
            .count();
        assert!(
            always_count <= 1,
            "Multiple All operands are not allowed (found {always_count})"
        );
        ArrayVec::new(self)
    }
}

// ============================================================================
// VeOperand - Unified operand type with automatic conversion
// ============================================================================

/// One operand type behind `impl Into<VeOperand<D, ..>>`, so a call site writes the constant
/// (`.vector_fxp(op, 16384i32)`), the borrowed VRF tensor (`&vrf`, read-many) or [`Stash`]
/// (read-once) and the conversion happens here.
#[derive(Debug)]
pub enum VeOperand<'a, D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> {
    /// Constant value (always valid).
    Const(D),
    /// VRF tensor reference.
    Vrf(&'a VrfTensor<D, Chip, Cluster, Slice, VrfMapping>),
    /// Stash operand.
    Stash(StashOperand<D>),
}

// From<i32> for VeOperand<i32, ...>
impl<Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<i32> for VeOperand<'_, i32, Chip, Cluster, Slice, VrfMapping> {
    fn from(v: i32) -> Self {
        VeOperand::Const(v)
    }
}

// From<f32> for VeOperand<f32, ...>
impl<Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<f32> for VeOperand<'_, f32, Chip, Cluster, Slice, VrfMapping> {
    fn from(v: f32) -> Self {
        VeOperand::Const(v)
    }
}

// From<&VrfTensor<D, ...>> for VeOperand<D, ...>
impl<'a, D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M>
    From<&'a VrfTensor<D, Chip, Cluster, Slice, VrfMapping>> for VeOperand<'a, D, Chip, Cluster, Slice, VrfMapping>
{
    fn from(vrf: &'a VrfTensor<D, Chip, Cluster, Slice, VrfMapping>) -> Self {
        VeOperand::Vrf(vrf)
    }
}

// From<StashOperand<D>> for VeOperand<D, ...>
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<StashOperand<D>>
    for VeOperand<'_, D, Chip, Cluster, Slice, VrfMapping>
{
    fn from(stash: StashOperand<D>) -> Self {
        VeOperand::Stash(stash)
    }
}

// From<Stash> for VeOperand<D, ...> — the stash read operand (symmetric with VeRhs).
impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> From<Stash>
    for VeOperand<'_, D, Chip, Cluster, Slice, VrfMapping>
{
    fn from(_: Stash) -> Self {
        VeOperand::Stash(StashOperand::always())
    }
}

impl<'a, D: VeScalar, Chip: M, Cluster: M, Slice: M, VrfMapping: M> VeOperand<'a, D, Chip, Cluster, Slice, VrfMapping> {
    /// Converts VeOperand to an ArrayVec of BinaryOperandTag with the target tensor mapping.
    pub fn into_branch_operands<Time: M, Packet: M>(
        self,
    ) -> ArrayVec<BinaryOperandTag<D, m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }]>, MAX_TAGS> {
        type TargetShape<Chip, Cluster, Slice, Time, Packet> =
            m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];

        match self {
            VeOperand::Const(v) => ArrayVec::new([BinaryOperandTag::always(VeRhs::Const { v })]),
            VeOperand::Vrf(vrf) => {
                let vrf_operand = VeRhs::<D, TargetShape<Chip, Cluster, Slice, Time, Packet>>::vrf(vrf);
                ArrayVec::new([BinaryOperandTag::always(vrf_operand)])
            }
            VeOperand::Stash(stash) => ArrayVec::new([BinaryOperandTag {
                operand0: VeRhs::Stash,
                operand1: (),
                tag_filter: stash.tag_filter,
            }]),
        }
    }
}

// ============================================================================
// IntoGroupOperand - Ergonomic operand conversion for VectorTensorPair
// ============================================================================

/// Optional per-group operand for VectorTensorPair operations. `None` skips the operation for
/// that group; `Some(operand)` applies it.
pub type GroupOperand<D, Mapping> = Option<BinaryOperandTag<D, Mapping>>;

/// Trait for converting various types into a [`GroupOperand`]. Accepts anything
/// `Into<BinaryOperandTag>` (i32/f32/`&VrfTensor`/etc.), `Option<BinaryOperandTag>` (pass-through),
/// or `()` to skip the operation for this group (`None`).
pub trait IntoGroupOperand<D: VeScalar, Mapping: M> {
    /// Converts into a [`GroupOperand`]. `None` skips the operation for this group.
    fn into_group_operand(self) -> GroupOperand<D, Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for () {
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        None
    }
}

/// `Option<BinaryOperandTag<D, Mapping>>` passes through.
impl<D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for Option<BinaryOperandTag<D, Mapping>> {
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        self
    }
}

/// Blanket impl: any type that implements `Into<BinaryOperandTag>` automatically implements
/// `IntoGroupOperand` by wrapping in `Some`. [`Plain`] is required because a pair group is always
/// [`Fresh`](crate::engine::vector::stash_slot::Fresh): pair mode has no stash to read, so `Stash`
/// here would lower to an operand nothing can fill.
impl<T, D: VeScalar, Mapping: M> IntoGroupOperand<D, Mapping> for T
where
    T: Into<BinaryOperandTag<D, Mapping>> + Plain,
{
    fn into_group_operand(self) -> GroupOperand<D, Mapping> {
        Some(self.into())
    }
}

// ============================================================================
// IntoGroupTernaryOperandTag - Ergonomic ternary operand conversion for VectorTensorPair
// ============================================================================

/// Type alias for group ternary operand in VectorTensorPair operations.
pub type GroupTernaryOperandTag<Mapping> = Option<TernaryOperandTag<Mapping>>;

/// Trait for converting various types into a group ternary operand. Accepts anything
/// `Into<TernaryOperandTag>` (`(f32, f32)`, `(VeRhs, f32)`, a tag), `Option<TernaryOperandTag>`
/// (pass-through), or `()` to skip the operation for this group.
pub trait IntoGroupTernaryOperandTag<Mapping: M> {
    /// Converts into a GroupTernaryOperandTag with the specified mapping.
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping>;
}

/// `()` represents skipping the operation for this group.
impl<Mapping: M> IntoGroupTernaryOperandTag<Mapping> for () {
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping> {
        None
    }
}

/// `Option<TernaryOperandTag<Mapping>>` passes through.
impl<Mapping: M> IntoGroupTernaryOperandTag<Mapping> for Option<TernaryOperandTag<Mapping>> {
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping> {
        self
    }
}

/// Blanket impl: any type that implements `Into<TernaryOperandTag>` automatically
/// implements `IntoGroupTernaryOperandTag` by wrapping in `Some`.
impl<T, Mapping: M> IntoGroupTernaryOperandTag<Mapping> for T
where
    T: Into<TernaryOperandTag<Mapping>>,
{
    fn into_group_ternary_operand(self) -> GroupTernaryOperandTag<Mapping> {
        Some(self.into())
    }
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::Broadcast;

    use super::*;

    type Map = Broadcast<1>;

    /// A `WithStash` two-branch list lowers to two per-group tags, each reading the stash, with the
    /// group filters intact. This is the value-level side of the read-once typestate: both branches
    /// name the stash, and the op reads it once for the pass (the read-once fold is checked at
    /// compile time by the `BranchOperands` doctests).
    #[test]
    fn multi_branch_stash_reads_lower_per_group() {
        let ops = BranchOperands::<f32, Map, _>::new()
            .stash_branch(GroupId::Zero)
            .stash_branch(GroupId::One);
        let tags: ArrayVec<BinaryOperandTag<f32, Map>, MAX_TAGS> = ops.into_operands();

        assert_eq!(tags.len(), 2);
        assert!(tags.iter().all(|t| matches!(t.operand0(), VeRhs::Stash)));
        assert!(matches!(tags[0].tag_filter(), TagFilter::Group { id: GroupId::Zero }));
        assert!(matches!(tags[1].tag_filter(), TagFilter::Group { id: GroupId::One }));
    }

    /// A stash branch mixed with a plain (const-rhs) branch: the plain branch keeps its constant,
    /// the stash branch reads the stash, and the whole list is still `WithStash` (reads once).
    #[test]
    fn mixed_stash_and_plain_branches_lower() {
        let ops = BranchOperands::<f32, Map, _>::new()
            .branch(2.0f32, GroupId::Zero)
            .stash_branch(GroupId::One);
        let tags: ArrayVec<BinaryOperandTag<f32, Map>, MAX_TAGS> = ops.into_operands();

        assert_eq!(tags.len(), 2);
        assert!(matches!(tags[0].operand0(), VeRhs::Const { v } if *v == 2.0));
        assert!(matches!(tags[1].operand0(), VeRhs::Stash));
    }
}
