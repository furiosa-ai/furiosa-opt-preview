//! Vector Engine tensor types for pipeline operations.
//!
//! This module provides tensor types that track VE pipeline stages at compile time
//! using the typestate pattern. Operations are expressed as method chaining, with
//! stage transitions enforced by the type system.
//!
//! # Key Types
//! - `VectorTensor`: Unified tensor type for all VE pipeline stages
//! - `VectorInitTensor`: VE input after `vector_init()`, before choosing the first block
//! - `VectorTensorPair`: Pair of tensors for two-group (interleaved) operations
//! - `VectorInterSliceReduceTensor`: After inter-slice reduce
//!
//! # Initialization / Entry (on CollectTensor / ContractTensor)
//! ```text
//! vector_init()                              → VectorInitTensor
//!   ├─ vector_intra_slice_tag(TagMode) → VectorBranchTensor           { VeOrder::IntraFirst }
//!   ├─ vector_intra_slice_unzip(...)         → VectorTensorPair             { VeOrder::IntraFirst }
//!   ├─ vector_inter_slice_reduce(op)         → VectorInterSliceReduceTensor { VeOrder::InterFirst }
//!   └─ vector_cast()                        → VectorCastTensor
//! ```
//!
//! # Exit
//! `vector_final()` exits the VE pipeline from any `Way8` stage and returns a `VectorFinalTensor`.
//! From there, `to_vrf()`, `to_vrf_view()`, `transpose()`, and `commit_trim()` are available.
//!
//! # Pipeline Paths
//! `intra` denotes intra-slice computation; `inter` denotes inter-slice reduction.
//! Brackets mark stages that may be skipped.
//! ```text
//! intra only:    vector_init → intra         → [filter] → [cast] → vector_final
//! intra → inter: vector_init → intra → inter → [filter] → [cast] → vector_final
//! inter only:    vector_init → inter         → [filter] → [cast] → vector_final
//! inter → intra: vector_init → inter → intra → [filter] → [cast] → vector_final
//! cast only:     vector_init → cast                             → vector_final
//! ```
//! Filter and cast sit outside both computation blocks: they read the last block's result,
//! whichever order the blocks ran in. The inter-slice reducer clears every tag bit, so a tag filter
//! needs the intra-slice block after it.
//!
//! # Intra-Slice Stage Flow (IntraSliceStage)
//! ```text
//! Tag → Logic → Fxp → FxpToFp → Narrow(Way4) → Fp → IntraSliceReduce → FpDiv → Widen(Way8) → FpToFxp → Clip
//! ```
//! Each stage's `Stage::OutputWay` is the packet width of the stream it emits: Narrow, Fp,
//! IntraSliceReduce and FpDiv emit `Way4`, and Widen emits `Way8` again.
//!
//! # Filter
//! `vector_filter_by_tag()` and `vector_filter_by_bitmap()` pack part of each slice into whole packets. They require `Way8` mode and the `Standalone` context.
//!
//! # VeOrder Tracking
//! The `const VE_ORDER: VeOrder` parameter tracks which unit was entered first:
//! - `IntraFirst` — after `vector_intra_slice_tag` / `vector_intra_slice_unzip`
//! - `InterFirst` — after `vector_inter_slice_reduce` from `VectorInitTensor`
//!
//! Once set, `VeOrder` is preserved through all subsequent operations.
//! It prevents re-entering the same VE block as an initial entry path.

mod vector_tensor;
mod vector_tensor_pair;
mod verify;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;
pub use vector_tensor::*;
pub use vector_tensor_pair::*;

use crate::backend::Backend;
use crate::constraints;
use crate::context::{Tu, TuContext};
use crate::engine::CanApplyVectorInit;
use crate::engine::vector::scalar::VeScalar;
use crate::runtime::CurrentBackend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::tensor::tu::{FixedPackets, Position, TimeExtent, TuTensor};

/// A VE stream's five-axis mapping, which the operand impls name to say an operand is read at the
/// stream's own partition. Public because those impls are, and their headers spell it out.
pub type VeTensorShape<Chip, Cluster, Slice, Time, Packet> = m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];

/// Stream position after `vector_final()` closes the Vector Engine pipeline.
#[derive(Debug)]
pub struct PositionVectorFinal;

impl Position for PositionVectorFinal {}

/// Tensor after the vector engine (after `vector_final()`).
pub type VectorFinalTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    B = CurrentBackend,
    E = FixedPackets,
> = TuTensor<'l, { T }, PositionVectorFinal, D, Chip, Cluster, Slice, Time, Packet, B, E>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend, E: TimeExtent>
    VectorFinalTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B, E>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        constraints::assert_packet_one_flit::<D, Packet>();
    }

    /// The Vector Engine's output with `time_extent`, as a filter leaves it.
    pub(crate) fn new_with_time_extent(
        device: &'l mut TuContext<{ T }>,
        inner: Tensor<D, Self::Mapping, B>,
        time_extent: E,
    ) -> Self {
        Self::check_constraints();

        Self {
            device,
            inner,
            time_extent,
            _position: PhantomData,
        }
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    VectorFinalTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(device: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::new_with_time_extent(device, inner, FixedPackets)
    }
}

//
// The Vector Engine accepts only `VeScalar` inputs (hardware constraint), so
// the bound lives on the impl rather than on a wider trait.
// ANCHOR: vector_init_impl
impl<'l, const T: Tu, P: CanApplyVectorInit, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Initializes Vector Engine processing for this tensor.
    #[primitive(TuTensor::vector_init)]
    pub fn vector_init(self) -> VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet> {
        VectorInitTensor::new(self.device, self.inner)
    }
}
// ANCHOR_END: vector_init_impl
