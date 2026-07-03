//! Packet Reducer: reduce-add within `Packet`.
//!
//! The Outer stage has already multiplied (and widened) the operands; the Packet
//! Reducer only sums along the contracted axes inside `Packet`.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::cast::ContractionCast;
use crate::context::*;
use crate::engine::contraction::outer::ContractOuterTensor;
use crate::engine::contraction::{ContractPacketTensor, TEMPORAL_ACCUMULATOR_COLS};
use crate::scalar::*;

// ANCHOR: contract_packet_def
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
    /// Spatial reduction within `Packet`: reduce-add along the contracted axes inside `Packet`
    /// of the product already produced by the Outer stage. `D` is the widened accumulator carried
    /// unchanged through this reduce; the DPE input packet is still sized in `Storage` bytes.
    #[primitive(ContractOuterTensor::contract_packet)]
    pub fn contract_packet<OutPacket: M>(
        self,
    ) -> ContractPacketTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, OutPacket, B> {
        verify_contract_packet::<Storage, Packet, OutPacket>();
        ContractPacketTensor::new(self.ctx, self.inner.reduce_add())
    }
}
// ANCHOR_END: contract_packet_def

/// Validates the Packet Reducer.
///
/// Checks:
/// 1. `Packet` should be 32 or 64 bytes (mirroring the Outer stage's `OutPacket`). The size is taken
///    in `Storage` (pre-widen) bytes: the DPE reads storage-width input flits, not accumulator-width.
/// 2. `OutPacket::SIZE` should be a power of two and at most
///    `TEMPORAL_ACCUMULATOR_COLS` and be obtainable from `Packet` by splitting
///    at a power-of-two sized boundary.
pub(crate) fn verify_contract_packet<Storage: Scalar, Packet: M, OutPacket: M>() {
    let packet_size = Storage::size_in_bytes_from_length(Packet::SIZE);
    assert!(
        [32, 64].contains(&packet_size),
        "Packet must be 32 or 64 bytes, got {packet_size} bytes"
    );

    assert!(
        OutPacket::SIZE <= TEMPORAL_ACCUMULATOR_COLS,
        "OutPacket::SIZE must be at most {TEMPORAL_ACCUMULATOR_COLS}, got {}",
        OutPacket::SIZE
    );

    assert!(
        OutPacket::SIZE.is_power_of_two(),
        "OutPacket::SIZE must be a power of two, got {}",
        OutPacket::SIZE
    );

    // `remove_padding` only peels the outer spine, so it needs a NORMALIZED layout: normalize first to
    // surface trailing padding to the top level, then strip it (a split can leave the pad buried inside a
    // strided composite like `(A # 16, C) / 4`, which a pre-normalize `remove_padding` would miss).
    let packet = Packet::to_value();
    let out_packet = OutPacket::to_value().normalize().remove_padding();

    assert!(
        (0..=Packet::SIZE.trailing_zeros()).rev().any(|depth| {
            let split = 1 << depth;
            packet.clone().split_at(split).0.normalize().remove_padding() == out_packet
        }),
        "OutPacket {out_packet} is not a valid contraction of Packet {packet}",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::bf16;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    #[test]
    fn valid_full_reduction() {
        verify_contract_packet::<i8, m![K], m![1]>();
    }

    #[test]
    fn valid_partial_reduction() {
        // K % 4 reduced
        verify_contract_packet::<i8, m![K], m![K / 4]>();
    }

    #[test]
    fn valid_partial_reduction_multi_axis() {
        // `D / 2 % 4` is reduced, retained_packet is `[A, D / 8]`.
        verify_contract_packet::<i8, m![A, D / 2], m![A, D / 8]>();
    }

    #[test]
    fn valid_padded_packet_inner_reduction() {
        verify_contract_packet::<i8, m![A # 16, C], m![A]>();
    }

    #[test]
    fn valid_padded_packet_inner_reduction_with_padding() {
        verify_contract_packet::<i8, m![A # 16, C], m![A # 16]>();
    }

    #[test]
    fn valid_padded_packet_split() {
        verify_contract_packet::<i8, m![B # 8, N], m![B]>();
    }

    #[test]
    fn valid_no_spatial_reduction_bf16() {
        // Tree depth 0: all 32 bf16 elements pass through, no reduction.
        verify_contract_packet::<bf16, m![D], m![D]>();
    }
}
