//! Pipeline stage definitions and transition traits for VE typestate pattern.
//!
//! # Typestate goals
//! - **Context markers** (`Standalone`, `Group`, `Zipped`) track which VE operations
//!   are valid for a tensor view. For example, `Zipped` tensors already represent the
//!   zipped result of a binary op, so they cannot `stash()` or `filter()` again.
//! - **Stage markers** (Tag, Logic, Fxp, …) mirror the hardware pipeline. Transition traits
//!   (`CanTransitionTo`) ensure the Rust type system disallows illegal stage ordering.
//! - **Way**: each stage names the packet width it receives (`Stage::InputWay`) and the width it
//!   emits (`Stage::OutputWay`), each `Way8` or `Way4`. An operation bounds the current stage's
//!   `OutputWay`, so a 4-way stream cannot exit or filter.
//! - **Commit gating**: only contexts that represent a full tensor (`Standalone`, `Zipped`)
//!   implement `Commitable`, preventing accidental commits of intermediate group states.
//!
//! # Pipeline order
//! Tag → Logic → Fxp → FxpToFp → Narrow → Fp → IntraSliceReduce → FpDiv → Widen → FpToFxp → Clip → Filter → Cast

use std::marker::ConstParamTy;

// ============================================================================
// Tensor context markers
// ============================================================================

/// Marker trait for VE tensor context state (`Standalone` / `Group` / `Zipped`), which
/// restricts the commit/stash/filter operations available on a tensor view.
pub trait VeTensorContext {}

/// Standalone tensor - represents a single tensor view that can use the full VE API
/// (stash/filter, stage transitions, commit).
#[derive(Debug, Clone, Copy)]
pub struct Standalone;
impl VeTensorContext for Standalone {}

/// In group context - used for individual groups managed by VectorTensorPair.
/// Only per-group operations (fxp, logic, clip, etc.) are allowed. Stash/common
/// operations are disabled so both groups stay in lock-step via the pair API.
#[derive(Debug, Clone, Copy)]
pub struct Group;
impl VeTensorContext for Group {}

/// After group zip - result of merging two groups. Filter/stash are NOT available
/// because VE already reduced to a single group.
#[derive(Debug, Clone, Copy)]
pub struct Zipped;
impl VeTensorContext for Zipped {}

/// Marker trait for contexts that can commit to `DmTensor`.
/// `Group` is intentionally excluded so users must zip the groups first.
pub trait Commitable: VeTensorContext {}

impl Commitable for Standalone {}

impl Commitable for Zipped {}

// ============================================================================
// VE entry order (IntraFirst)
// ============================================================================

/// Tracks which VE block was entered first.
///
/// - `IntraFirst`: intra-slice chain (intra-slice) was entered first via `vector_intra_slice_tag` or `vector_intra_slice_unzip`.
/// - `InterFirst`: inter-slice reducer (inter-slice) was entered first via `vector_inter_slice_reduce` on `VectorInitTensor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ConstParamTy)]
pub enum VeOrder {
    /// intra-slice chain was entered first.
    IntraFirst,
    /// inter-slice reducer was entered first.
    InterFirst,
}

// ============================================================================
// Way markers (4-way / 8-way)
// ============================================================================

/// `Way` — the packet width of the stream a stage emits, 8 or 4 elements; each [`Stage`] names one.
///
/// The hardware VE flit is 8 elements wide, but float ALU only processes the front 4.
/// `Way4` stages hold packets narrowed to 4 elements via `vector_narrow_split` or `vector_narrow_trim`.
/// Float operations are only available in `Way4`, enforced at compile time.
///
/// A stream leaves the Vector Engine only from a `Way8` stage:
///
/// ```
/// # #![feature(adt_const_params)]
/// use furiosa_opt_std::prelude::*;
/// axes![S = 64, P = 8];
/// fn exit<'l>(input: CollectTensor<'l, { Tu::Main }, f32, m![1], m![1], m![S], m![1], m![P]>) {
///     let _ = input.vector_init().vector_intra_slice_tag(TagMode::Zero).vector_final();
/// }
/// ```
///
/// ```compile_fail,E0599
/// # #![feature(adt_const_params)]
/// use furiosa_opt_std::prelude::*;
/// axes![S = 64, P = 8];
/// fn exit<'l>(input: CollectTensor<'l, { Tu::Main }, f32, m![1], m![1], m![S], m![1], m![P]>) {
///     let _ = input
///         .vector_init()
///         .vector_intra_slice_tag(TagMode::Zero)
///         .vector_narrow_split::<m![P / 4], m![P % 4]>()
///         .vector_final();
/// }
/// ```
pub trait Way: std::fmt::Debug {
    /// Valid packet elements per flit: 8 for `Way8`, 4 for `Way4`.
    const PACKET_ELEMENTS: usize;
}

/// 8-way: full 8-element flit. Transition to `Way4` via `vector_narrow_split`
/// (>4 real elements) or `vector_narrow_trim` (≤4).
#[derive(Debug, Clone, Copy)]
pub struct Way8;
impl Way for Way8 {
    const PACKET_ELEMENTS: usize = 8;
}

/// 4-way: front-4-only flit. Transition back to `Way8` via `vector_widen_concat` or
/// `vector_widen_pad`.
#[derive(Debug, Clone, Copy)]
pub struct Way4;
impl Way for Way4 {
    const PACKET_ELEMENTS: usize = 4;
}

// ============================================================================
// Stage marker types
// ============================================================================

/// Trait for pipeline stage markers.
pub trait Stage {
    /// The packet width of the stream entering this stage.
    type InputWay: Way;
    /// The packet width of the stream this stage emits. Only `Narrow` (8 to 4) and `Widen` (4 to 8)
    /// differ from `InputWay`.
    type OutputWay: Way;
}

/// Trait for intra-slice chain (intra-slice) stages: Tag through Clip.
pub trait IntraSliceStage: Stage {}

/// Trait for inter-slice reducer (inter-slice) stages.
pub trait InterSliceStage: Stage {}

/// Trait for stages that support stash operation.
/// Only specific stages can stash current data (based on hardware ALU availability).
/// Supported: Tag, Logic, Fxp, Narrow, Fp, FpDiv, Clip
/// NOT supported: FxpToFp, IntraSliceReduce, Widen, FpToFxp
pub trait Stashable: Stage {}

/// Tag unit configuration stage.
#[derive(Debug, Clone, Copy)]
pub struct Tag;
impl Stage for Tag {
    type InputWay = Way8;
    type OutputWay = Way8;
}
impl IntraSliceStage for Tag {}
impl Stashable for Tag {}

/// Logic cluster stage.
#[derive(Debug, Clone, Copy)]
pub struct Logic;
impl Logic {
    /// Maximum number of operations in Logic cluster.
    pub const MAX_OPS: usize = 5;
}
impl Stage for Logic {
    type InputWay = Way8;
    type OutputWay = Way8;
}
impl IntraSliceStage for Logic {}
impl Stashable for Logic {}

/// Fixed-point cluster stage.
#[derive(Debug, Clone, Copy)]
pub struct Fxp;
impl Fxp {
    /// Maximum number of operations in Fxp cluster.
    pub const MAX_OPS: usize = 4;
}
impl Stage for Fxp {
    type InputWay = Way8;
    type OutputWay = Way8;
}
impl IntraSliceStage for Fxp {}
impl Stashable for Fxp {}

/// FxpToFp conversion stage.
#[derive(Debug, Clone, Copy)]
pub struct FxpToFp;
impl Stage for FxpToFp {
    type InputWay = Way8;
    type OutputWay = Way8;
}
impl IntraSliceStage for FxpToFp {}

/// Narrow layer stage.
#[derive(Debug, Clone, Copy)]
pub struct Narrow;
impl Stage for Narrow {
    type InputWay = Way8;
    type OutputWay = Way4;
}
impl IntraSliceStage for Narrow {}
impl Stashable for Narrow {}

/// Fp cluster stage.
#[derive(Debug, Clone, Copy)]
pub struct Fp;
impl Fp {
    /// Maximum number of operations in Fp cluster.
    pub const MAX_OPS: usize = 5;
}
impl Stage for Fp {
    type InputWay = Way4;
    type OutputWay = Way4;
}
impl IntraSliceStage for Fp {}
impl Stashable for Fp {}

/// Intra-slice reduce stage.
#[derive(Debug, Clone, Copy)]
pub struct IntraSliceReduce;
impl Stage for IntraSliceReduce {
    type InputWay = Way4;
    type OutputWay = Way4;
}
impl IntraSliceStage for IntraSliceReduce {}

/// FpDiv stage.
#[derive(Debug, Clone, Copy)]
pub struct FpDiv;
impl Stage for FpDiv {
    type InputWay = Way4;
    type OutputWay = Way4;
}
impl IntraSliceStage for FpDiv {}
impl Stashable for FpDiv {}

/// Widen layer stage.
#[derive(Debug, Clone, Copy)]
pub struct Widen;
impl Stage for Widen {
    type InputWay = Way4;
    type OutputWay = Way8;
}
impl IntraSliceStage for Widen {}

/// FpToFxp conversion stage.
#[derive(Debug, Clone, Copy)]
pub struct FpToFxp;
impl Stage for FpToFxp {
    type InputWay = Way8;
    type OutputWay = Way8;
}
impl IntraSliceStage for FpToFxp {}

/// Clip cluster stage.
#[derive(Debug, Clone, Copy)]
pub struct Clip;
impl Clip {
    /// Maximum number of operations in Clip cluster.
    pub const MAX_OPS: usize = 3;
}
impl Stage for Clip {
    type InputWay = Way8;
    type OutputWay = Way8;
}
impl IntraSliceStage for Clip {}
impl Stashable for Clip {}

/// Inter-slice reduce stage — after inter-slice reduce operation.
#[derive(Debug, Clone, Copy)]
pub struct InterSliceReduce;
impl Stage for InterSliceReduce {
    type InputWay = Way8;
    type OutputWay = Way8;
}
impl InterSliceStage for InterSliceReduce {}

/// Filter stage (applies branch filtering before output).
#[derive(Debug, Clone, Copy)]
pub struct Filter;
impl Stage for Filter {
    type InputWay = Way8;
    type OutputWay = Way8;
}

/// Vector Engine cast stage.
#[derive(Debug, Clone, Copy)]
pub struct Cast;
impl Stage for Cast {
    type InputWay = Way8;
    type OutputWay = Way8;
}

// ============================================================================
// Stage transition traits
// ============================================================================

/// Generic marker trait for stage transitions.
/// `CanTransitionTo<Target>` indicates that a stage can transition to `Target`, and requires the
/// stage to emit the width `Target` receives.
pub trait CanTransitionTo<Target: Stage>: Stage<OutputWay = Target::InputWay> {}

// Each block lists the stages that may move into one stage.

// Tag transitions (re-entering the intra-slice chain after the inter-slice reducer)
impl CanTransitionTo<Tag> for InterSliceReduce {}

// Logic transitions
impl CanTransitionTo<Logic> for Tag {}
impl CanTransitionTo<Logic> for Logic {} // stay in Logic

// Fxp transitions
impl CanTransitionTo<Fxp> for Tag {}
impl CanTransitionTo<Fxp> for Logic {}
impl CanTransitionTo<Fxp> for Fxp {} // stay in Fxp

// FxpToFp transitions
impl CanTransitionTo<FxpToFp> for Tag {}
impl CanTransitionTo<FxpToFp> for Logic {}
impl CanTransitionTo<FxpToFp> for Fxp {}

// Narrow transitions (Way8 → Way4)
impl CanTransitionTo<Narrow> for Tag {}
impl CanTransitionTo<Narrow> for Logic {}
impl CanTransitionTo<Narrow> for Fxp {}
impl CanTransitionTo<Narrow> for FxpToFp {}

// Fp transitions
impl CanTransitionTo<Fp> for Narrow {}
impl CanTransitionTo<Fp> for Fp {} // stay in Fp

// IntraSliceReduce transitions
impl CanTransitionTo<IntraSliceReduce> for Narrow {}
impl CanTransitionTo<IntraSliceReduce> for Fp {}

// FpDiv transitions
impl CanTransitionTo<FpDiv> for Narrow {}
impl CanTransitionTo<FpDiv> for Fp {}
impl CanTransitionTo<FpDiv> for IntraSliceReduce {}

// Widen transitions (Way4 → Way8)
impl CanTransitionTo<Widen> for Narrow {}
impl CanTransitionTo<Widen> for Fp {}
impl CanTransitionTo<Widen> for IntraSliceReduce {}
impl CanTransitionTo<Widen> for FpDiv {}

// FpToFxp transitions
impl CanTransitionTo<FpToFxp> for Tag {}
impl CanTransitionTo<FpToFxp> for Logic {}
impl CanTransitionTo<FpToFxp> for Fxp {}
impl CanTransitionTo<FpToFxp> for FxpToFp {}
impl CanTransitionTo<FpToFxp> for Widen {}

// Clip transitions
impl CanTransitionTo<Clip> for Tag {}
impl CanTransitionTo<Clip> for Logic {}
impl CanTransitionTo<Clip> for Fxp {}
impl CanTransitionTo<Clip> for FxpToFp {}
impl CanTransitionTo<Clip> for Widen {}
impl CanTransitionTo<Clip> for FpToFxp {}
impl CanTransitionTo<Clip> for Clip {} // stay in Clip

// InterSliceReduce transitions (from the intra-slice chain)
// Tag is intentionally excluded: callers should use VectorInitTensor::vector_inter_slice_reduce
// directly instead of branching first and then immediately transitioning.
impl CanTransitionTo<InterSliceReduce> for Logic {}
impl CanTransitionTo<InterSliceReduce> for Fxp {}
impl CanTransitionTo<InterSliceReduce> for FxpToFp {}
impl CanTransitionTo<InterSliceReduce> for Widen {}
impl CanTransitionTo<InterSliceReduce> for FpToFxp {}
impl CanTransitionTo<InterSliceReduce> for Clip {}

// Filter transitions (after either computation block)
impl CanTransitionTo<Filter> for Tag {}
impl CanTransitionTo<Filter> for Logic {}
impl CanTransitionTo<Filter> for Fxp {}
impl CanTransitionTo<Filter> for FxpToFp {}
impl CanTransitionTo<Filter> for Widen {}
impl CanTransitionTo<Filter> for FpToFxp {}
impl CanTransitionTo<Filter> for Clip {}
impl CanTransitionTo<Filter> for InterSliceReduce {}

/// Stages whose stream still carries the tags `vector_intra_slice_tag` set, which a tag filter
/// reads. The inter-slice reducer clears every tag bit, so a tag filter cannot follow it.
pub trait CarriesTag: CanTransitionTo<Filter> {}
impl CarriesTag for Tag {}
impl CarriesTag for Logic {}
impl CarriesTag for Fxp {}
impl CarriesTag for FxpToFp {}
impl CarriesTag for Widen {}
impl CarriesTag for FpToFxp {}
impl CarriesTag for Clip {}

// Cast transitions (after either computation block; a filter's output casts on its own position)
impl CanTransitionTo<Cast> for Tag {}
impl CanTransitionTo<Cast> for Logic {}
impl CanTransitionTo<Cast> for Fxp {}
impl CanTransitionTo<Cast> for FxpToFp {}
impl CanTransitionTo<Cast> for Widen {}
impl CanTransitionTo<Cast> for FpToFxp {}
impl CanTransitionTo<Cast> for Clip {}
impl CanTransitionTo<Cast> for InterSliceReduce {}
