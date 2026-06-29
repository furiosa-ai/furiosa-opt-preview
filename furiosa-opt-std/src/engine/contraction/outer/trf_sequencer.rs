//! TRF Sequencer: TRF-side validation and transpose for [`super::contract_outer`].
//!
//! - [`verify_trf_sequencer`]: checks that `TrfElement` expands into the joint
//!   `OutTime`/`OutPacket` mapping (TRF-side counterpart of
//!   [`super::stream_adapter::verify_stream_adapter`]).
//! - [`contract_outer`]: transposes the TRF tensor into the joint shape.

use furiosa_mapping::*;

use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::tensor::memory::TrfTensor;

pub(super) fn contract_outer<
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    TrfElement: M,
    OutTime: M,
    OutPacket: M,
    B: Backend,
>(
    trf_tensor: &TrfTensor<D, Chip, Cluster, Slice, Lane, TrfElement, B>,
) -> Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { OutTime }, { OutPacket }], B> {
    verify_trf_sequencer::<D, Lane, TrfElement, OutTime, OutPacket>();
    trf_tensor.inner.transpose(true)
}

/// Validates the TRF-side mapping transformation for [`super::contract_outer`]:
/// that `TrfElement` (broadcast across `Lane`) expands into the joint
/// `OutTime`/`OutPacket` mapping, the counterpart of
/// [`super::stream_adapter::verify_stream_adapter`].
#[allow(clippy::extra_unused_type_parameters)] // TODO: parameters are consumed once the check below lands.
fn verify_trf_sequencer<D: Scalar, Lane: M, TrfElement: M, OutTime: M, OutPacket: M>() {
    // TODO: validate the TRF-side expansion (TrfElement -> OutTime/OutPacket),
    // mirroring the stream-side checks in `verify_stream_adapter`.
}
