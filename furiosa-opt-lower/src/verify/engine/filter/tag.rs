//! The tag filter: each slice emits the packets its survivors fill.

use furiosa_mapping::{Mapping, MappingExt};

use super::{FilterError, FilterInput};
use crate::verify::ONE_FLIT_ELEMENTS;

/// Validates a tag filter run on the chips and clusters of `clusters`: the output is whole packets
/// with the valid elements first.
pub fn config_filter_by_tag(layouts: FilterInput, clusters: &Mapping) -> Result<(), FilterError> {
    layouts.check_packets()?;
    // Padding carries a tag like any element, so a guard could let it through.
    if layouts.input().normalize().has_padding() {
        return Err(FilterError::TagFilterInputPadding {
            layouts: Box::new(layouts),
        });
    }
    // The output is its elements, then padding that completes the last packet and adds none of
    // its own. Every axis holds at least one element, so `valid` is never zero.
    let valid = layouts.leading_valid()?;
    if layouts.output().size() != valid.next_multiple_of(ONE_FLIT_ELEMENTS) {
        return Err(FilterError::OutputVolume {
            kept_elements: valid,
            layouts: Box::new(layouts),
        });
    }
    // Survivors move by their order, not by an axis, so no input axis can name them.
    if !layouts.names_only_new_axes(&layouts.output()) {
        return Err(FilterError::TagOutputNamesInputAxes {
            layouts: Box::new(layouts),
        });
    }
    // A slice emits at most the packets it reads.
    if layouts.out_time.size() > layouts.in_time.size() {
        return Err(FilterError::OutputCapacity {
            layouts: Box::new(layouts),
        });
    }
    // A padded chip or cluster holds undefined values, which may pass the guard and overflow the
    // output, and the commit cannot skip a chip or a cluster as it skips a `#{!}` slice. An output
    // that holds every packet a slice reads cannot overflow.
    // TODO: Once a commit skips a `#{!}` chip or cluster, check this in `config_commit_slice` with
    // the slices instead.
    if clusters.clone().normalize().has_padding() && layouts.out_time.size() < layouts.in_time.size() {
        return Err(FilterError::TagFilterPaddedCluster {
            clusters: clusters.clone(),
            layouts: Box::new(layouts),
        });
    }
    Ok(())
}

impl FilterInput {
    /// Valid elements of the output, which come first with only padding after.
    fn leading_valid(&self) -> Result<usize, FilterError> {
        let output = self.output();
        let leading = output.normalize().remove_outermost_padding().normalize();
        match leading.has_no_padding() {
            true => Ok(leading.size()),
            false => Err(FilterError::InteriorOutputPadding {
                layouts: Box::new(self.clone()),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![T = 2, P = 8, U = 2, Q = 8, CL = 1];

    #[test]
    fn a_tag_filter_output_of_only_padding_packets_is_refused() {
        // A packet of padding alone holds nothing a slice may emit into.
        assert!(matches!(
            config_filter_by_tag(
                FilterInput {
                    in_time: <m![T]>::to_value(),
                    in_packet: <m![P]>::to_value(),
                    out_time: <m![U # 4]>::to_value(),
                    out_packet: <m![Q]>::to_value()
                },
                &Mapping::identity()
            ),
            Err(FilterError::OutputVolume { kept_elements: 16, .. })
        ));
    }

    #[test]
    fn a_tag_filter_output_larger_than_its_input_is_refused() {
        // A slice reads one packet, so it emits at most one into an output of two.
        assert!(matches!(
            config_filter_by_tag(
                FilterInput {
                    in_time: Mapping::identity(),
                    in_packet: <m![P]>::to_value(),
                    out_time: <m![U]>::to_value(),
                    out_packet: <m![Q]>::to_value()
                },
                &Mapping::identity()
            ),
            Err(FilterError::OutputCapacity { .. })
        ));
    }

    #[test]
    fn a_tag_filter_rejects_padded_input_and_misplaced_output_padding() {
        let packet = <m![P]>::to_value();
        assert!(matches!(
            config_filter_by_tag(
                FilterInput {
                    in_time: <m![T # 4]>::to_value(),
                    in_packet: packet.clone(),
                    out_time: Mapping::identity(),
                    out_packet: packet.clone()
                },
                &Mapping::identity()
            ),
            Err(FilterError::TagFilterInputPadding { .. })
        ));
        assert!(matches!(
            config_filter_by_tag(
                FilterInput {
                    in_time: Mapping::identity(),
                    in_packet: <m![P % 4 # 8]>::to_value(),
                    out_time: Mapping::identity(),
                    out_packet: packet.clone()
                },
                &Mapping::identity()
            ),
            Err(FilterError::TagFilterInputPadding { .. })
        ));
        assert!(matches!(
            config_filter_by_tag(
                FilterInput {
                    in_time: Mapping::identity(),
                    in_packet: <m![P % 4]>::to_value(),
                    out_time: Mapping::identity(),
                    out_packet: packet.clone()
                },
                &Mapping::identity()
            ),
            Err(FilterError::PacketSize { .. })
        ));
        // Padding between the valid elements leaves no leading capacity.
        axes![X = 2];
        assert!(matches!(
            config_filter_by_tag(
                FilterInput {
                    in_time: <m![T]>::to_value(),
                    in_packet: packet.clone(),
                    out_time: Mapping::identity(),
                    out_packet: <m![X, 1 # 4]>::to_value()
                },
                &Mapping::identity()
            ),
            Err(FilterError::InteriorOutputPadding { .. })
        ));
    }

    #[test]
    fn a_tag_filter_output_names_its_survivors_with_new_axes() {
        let input = |out_time: Mapping, out_packet: Mapping| FilterInput {
            in_time: <m![T]>::to_value(),
            in_packet: <m![P]>::to_value(),
            out_time,
            out_packet,
        };
        assert_eq!(
            config_filter_by_tag(input(<m![U]>::to_value(), <m![Q]>::to_value()), &Mapping::identity()),
            Ok(())
        );
        // Position k of the output holds the k-th survivor, not the input's element at k.
        assert!(matches!(
            config_filter_by_tag(input(<m![T]>::to_value(), <m![P]>::to_value()), &Mapping::identity()),
            Err(FilterError::TagOutputNamesInputAxes { .. })
        ));
        assert!(matches!(
            config_filter_by_tag(input(<m![U]>::to_value(), <m![P]>::to_value()), &Mapping::identity()),
            Err(FilterError::TagOutputNamesInputAxes { .. })
        ));
    }

    #[test]
    fn a_tag_filter_on_padded_clusters_holds_every_packet_a_slice_reads() {
        let input = |out_time: Mapping| FilterInput {
            in_time: <m![T]>::to_value(),
            in_packet: <m![P]>::to_value(),
            out_time,
            out_packet: <m![Q]>::to_value(),
        };
        let padded = <m![CL # 2]>::to_value();
        // Cluster 1 holds undefined values, but an output of every packet cannot overflow.
        assert_eq!(config_filter_by_tag(input(<m![U]>::to_value()), &padded), Ok(()));
        assert!(matches!(
            config_filter_by_tag(input(Mapping::identity()), &padded),
            Err(FilterError::TagFilterPaddedCluster { .. })
        ));
        assert_eq!(
            config_filter_by_tag(input(Mapping::identity()), &<m![CL]>::to_value()),
            Ok(())
        );
    }
}
