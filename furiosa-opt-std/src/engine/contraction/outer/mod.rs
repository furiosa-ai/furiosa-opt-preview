//! Outer stage -- broadcasts both operands and multiplies them elementwise.
//!
//! The Outer stage runs three sub-stages in series:
//!
//! - `stream_adapter`: validates the stream (`CollectTensor`) side and
//!   transposes it into the joint computation mapping.
//! - `trf_sequencer`: transposes the TRF into the joint computation mapping.
//! - Multiplier: elementwise multiplies the two aligned operands. Inputs widen
//!   to the contraction output type (`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`)
//!   before multiplication, matching the DPE accumulator type.
//!
//! The resulting [`ContractOuterTensor`] carries a *single* multiplied tensor in
//! the common computation mapping `[Chip, Cluster, Slice, Lane, Time, Packet]`,
//! ready for the [`super::packet`] reduce-add.

pub(super) mod stream_adapter;
pub(super) mod trf_sequencer;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::{Cast, ContractionCast};
use crate::constraints;
use crate::context::*;
use crate::engine::CanApplyContractOuter;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::memory::TrfTensor;
use crate::tensor::tu::TuTensor;

/// Output of the Outer stage: the multiplied product (Stream Adapter * TRF Sequencer), ready for the Packet Reducer.
///
/// The product is already cast to the contraction output type ([`ContractionCast::Output`]),
/// so the downstream Packet Reducer only has to run reduce-add.
#[derive(Debug)]
pub struct ContractOuterTensor<
    'l,
    const T: Tu,
    D: Scalar + ContractionCast,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<
        <D as ContractionCast>::Output,
        Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>,
        B,
    >,
}

impl<
    'l,
    const T: Tu,
    D: Scalar + ContractionCast,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractOuterTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    const fn check_constraints() {
        constraints::assert_cluster_size(Cluster::SIZE);
        constraints::assert_slice_size(Slice::SIZE);
        constraints::assert_packet_one_or_two_flit(D::BITS, Packet::SIZE);
    }

    #[doc(hidden)]
    pub(crate) fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<
            <D as ContractionCast>::Output,
            Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>,
            B,
        >,
    ) -> Self {
        const { Self::check_constraints() };
        Self { ctx, inner }
    }
}

// ANCHOR: contract_outer_def
impl<
    'l,
    const T: Tu,
    P: CanApplyContractOuter,
    D: Scalar + ContractionCast,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Outer stage: Stream Adapter + TRF Sequencer broadcast both operands to a matching
    /// computation shape, then the Multiplier widens to the contraction output type and multiplies
    /// them elementwise. The result is a single multiplied tensor ready for the Packet Reducer.
    #[primitive(TuTensor::contract_outer)]
    pub fn contract_outer<OutTime: M, OutPacket: M, Lane: M, TrfElement: M>(
        self,
        trf_tensor: &TrfTensor<D, Chip, Cluster, Slice, Lane, TrfElement, B>,
    ) -> ContractOuterTensor<'l, T, D, Chip, Cluster, Slice, Lane, OutTime, OutPacket, B>
    where
        D: Cast<<D as ContractionCast>::Output>,
    {
        // Broadcast both operands into the joint computation mapping. Each side
        // verifies its own mapping inside its `contract_outer`.
        let lhs = stream_adapter::contract_outer::<D, Chip, Cluster, Slice, Lane, Time, Packet, OutTime, OutPacket, B>(
            self.inner,
        );
        let trf = trf_sequencer::contract_outer::<D, Chip, Cluster, Slice, Lane, TrfElement, OutTime, OutPacket, B>(
            trf_tensor,
        );

        // Multiplier: widen to D::Output, then elementwise multiply.
        let lhs = lhs.map(|v| v.cast());
        let trf = trf.map(|v| v.cast());
        let inner = lhs.zip_with(&trf, |a, b| a * b);

        ContractOuterTensor::new(self.ctx, inner)
    }
}
// ANCHOR_END: contract_outer_def
