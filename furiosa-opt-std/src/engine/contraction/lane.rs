//! Lane Folder (`contract_lane`): folds `Lane` into the output stream,
//! producing the contraction pipeline's final [`ContractTensor`].

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::context::*;
use crate::engine::contraction::{ContractTensor, ContractTimeTensor};
use crate::scalar::*;

/// Contraction mode for the Lane Folder.
#[primitive(LaneMode)]
#[derive(Clone, Debug)]
pub enum LaneMode {
    /// Interleaved: outputs data element-by-element across all `Lane`s.
    Interleaved,
    /// Sequential: outputs reduced data in each `Lane` sequentially.
    Sequential,
}

impl std::fmt::Display for LaneMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaneMode::Interleaved => write!(f, "Interleaved"),
            LaneMode::Sequential => write!(f, "Sequential"),
        }
    }
}

// ANCHOR: contract_lane_def
impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Folds the `Lane` dimension into the output stream.
    /// `LaneMode::Interleaved` relocates `Lane` into `OutPacket`;
    /// `LaneMode::Sequential` relocates `Lane` into `OutTime`.
    #[primitive(ContractTimeTensor::contract_lane)]
    pub fn contract_lane<OutTime: M, OutPacket: M>(
        self,
        mode: LaneMode,
    ) -> ContractTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_contract_lane(
            Lane::to_value(),
            Time::to_value(),
            Packet::to_value(),
            OutTime::to_value(),
            OutPacket::to_value(),
            self.pre_reduce_time,
            mode,
        );
        ContractTensor::new(self.ctx, self.inner.transpose(false))
    }
}
// ANCHOR_END: contract_lane_def

/// Validates the Lane Folder.
///
/// Checks:
/// Validates the Lane Folder via [`furiosa_opt_lower::config_contract_lane`] (packet / time /
/// accumulator rules documented there).
pub(crate) fn verify_contract_lane(
    lane: Mapping,
    time: Mapping,
    packet: Mapping,
    out_time: Mapping,
    out_packet: Mapping,
    pre_reduce_time: Mapping,
    kind: LaneMode,
) {
    let mode = match kind {
        LaneMode::Interleaved => furiosa_opt_lower::LaneMode::Interleaved,
        LaneMode::Sequential => furiosa_opt_lower::LaneMode::Sequential,
    };
    furiosa_opt_lower::config_contract_lane(&lane, &time, &packet, &out_time, &out_packet, &pre_reduce_time, mode)
        .unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![A = 4, B = 2, C = 4, D = 32, K = 64, M = 4, N = 8, O = 2, P = 8];

    mod out_packet_size {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![A]>::to_value(),
                <m![1]>::to_value(),
                <m![A]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![A]>::to_value(),
                LaneMode::Interleaved,
            );
        }
    }

    mod interleaved {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_padding() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![B # 4]>::to_value(),
                <m![1]>::to_value(),
                <m![B # 4]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B # 4]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            verify_contract_lane(
                <m![1]>::to_value(),
                <m![A # 8, B]>::to_value(),
                <m![D]>::to_value(),
                <m![A # 8, B, D]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![A # 8, B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_non_outermost() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![C, B]>::to_value(),
                <m![1]>::to_value(),
                <m![C, B]>::to_value(),
                <m![N]>::to_value(),
                <m![C, B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_four_rows() {
            verify_contract_lane(
                <m![M]>::to_value(),
                <m![C, B]>::to_value(),
                <m![1]>::to_value(),
                <m![C, B]>::to_value(),
                <m![M # 8]>::to_value(),
                <m![C, B]>::to_value(),
                LaneMode::Interleaved,
            );
        }

        #[test]
        fn valid_all_time_reduced() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                <m![1]>::to_value(),
                <m![1]>::to_value(),
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                LaneMode::Interleaved,
            );
        }
    }

    mod sequential {
        use super::*;
        use furiosa_mapping::M as _;

        #[test]
        fn valid() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B, N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_padded_row() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B, N # 8]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_all_time_reduced() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                <m![1]>::to_value(),
                <m![N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![1]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_no_reduction_with_padding() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![A # 8, B]>::to_value(),
                <m![1]>::to_value(),
                <m![A # 8, B, N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![A # 8, B]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_padded_packet() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![M]>::to_value(),
                <m![B]>::to_value(),
                <m![M, N]>::to_value(),
                <m![B # 8]>::to_value(),
                <m![M]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_full_temporal_reduction() {
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![1]>::to_value(),
                <m![D]>::to_value(),
                <m![N, D / 8]>::to_value(),
                <m![D % 8]>::to_value(),
                <m![1]>::to_value(),
                LaneMode::Sequential,
            );
        }

        #[test]
        fn valid_multi_axis_reduction() {
            // Reduce A and C. post=[B].
            verify_contract_lane(
                <m![N]>::to_value(),
                <m![B]>::to_value(),
                <m![1]>::to_value(),
                <m![B, N]>::to_value(),
                <m![1 # 8]>::to_value(),
                <m![B]>::to_value(),
                LaneMode::Sequential,
            );
        }
    }
}
