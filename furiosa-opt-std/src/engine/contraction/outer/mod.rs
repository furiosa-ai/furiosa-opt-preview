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
//!
//! The widen happens *entering* Outer: `contract_outer` takes storage-typed operands `D` and yields a
//! [`ContractOuterTensor`] whose element type is already the widened accumulator
//! [`ContractionCast::Output`]. That widened type is then the tensor's *own* type parameter, so every
//! downstream stage (Packet Reducer, Time Reducer, Lane Folder) carries the accumulator width verbatim,
//! with no per-stage `ContractionCast` re-derivation. The narrow back to a storage dtype is not a stage
//! here at all — it happens naturally at the next [`crate::engine::cast::CastTensor`] the consumer applies.

pub(super) mod stream_adapter;
pub(super) mod trf_sequencer;

use std::marker::PhantomData;

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
/// Two dtype params, related by the `Storage: ContractionCast<Output = D>` bound that pins them:
/// - `D` is the *widened accumulator* the product is carried in (`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`).
/// - `Storage` is the pre-widen operand dtype. It is retained only so the Packet Reducer's physical-flit
///   check ([`super::packet::verify_contract_packet`]) can size the DPE *input* packet in storage bytes,
///   which the widened `D` no longer encodes (`Storage` is not recoverable from `D` — the widen is
///   many-to-one, so this second param is load-bearing, not redundant).
///
/// See the module docs for where the widen and the (downstream) narrow happen.
#[derive(Debug)]
pub struct ContractOuterTensor<
    'l,
    const T: Tu,
    D: Scalar,
    Storage: ContractionCast<Output = D>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
    /// Records the pre-widen storage dtype for the flit check only; never materialized. See struct doc.
    pub(crate) _storage: PhantomData<Storage>,
}

impl<
    'l,
    const T: Tu,
    D: Scalar,
    Storage: ContractionCast<Output = D>,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend,
> ContractOuterTensor<'l, T, D, Storage, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        // The DPE reads storage-width input flits, then widens; size the packet in `Storage` bytes.
        constraints::assert_packet_one_or_two_flit::<Storage, Packet>();
    }

    #[doc(hidden)]
    pub(crate) fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
    ) -> Self {
        Self::check_constraints();
        Self {
            ctx,
            inner,
            _storage: PhantomData,
        }
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
    ) -> ContractOuterTensor<'l, T, <D as ContractionCast>::Output, D, Chip, Cluster, Slice, Lane, OutTime, OutPacket, B>
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

        // Multiplier: widen each operand to D::Output and multiply in ONE fused zip pass (vs two
        // `map` widen passes + a `zip_with` multiply: three full-product buffers collapsed to one).
        // Parity-identical -- same per-cell `cast` then `*` -- it just skips materializing the two
        // intermediate widened operands.
        let inner = lhs.zip_with(&trf, |a, b| {
            let (a, b): (<D as ContractionCast>::Output, <D as ContractionCast>::Output) = (a.cast(), b.cast());
            a * b
        });

        ContractOuterTensor::new(self.ctx, inner)
    }
}
// ANCHOR_END: contract_outer_def
