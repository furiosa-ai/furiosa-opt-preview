//! Output placement rules for bitmap filters.

use furiosa_mapping::{Index, Mapping, MappingExt, PaddingKind};
use furiosa_opt_common_ir::CollectBitmap;

use super::{FilterError, FilterInput};
use crate::verify::ONE_FLIT_ELEMENTS;

const _: () = assert!(ONE_FLIT_ELEMENTS == CollectBitmap::BITS as usize);

/// Validates the output placement for the positions selected by `bitmap`.
pub fn config_filter_by_bitmap(layouts: FilterInput, bitmap: CollectBitmap) -> Result<(), FilterError> {
    let positions = bitmap.positions().collect::<Vec<_>>();
    if positions.is_empty() {
        return Err(FilterError::EmptyBitmap);
    }
    layouts.check_packets()?;
    let kept_elements = layouts.in_time.size() * positions.len();
    if layouts.output().size() != kept_elements.next_multiple_of(ONE_FLIT_ELEMENTS) {
        return Err(FilterError::OutputVolume {
            kept_elements,
            layouts: Box::new(layouts),
        });
    }
    let spelling = input_axes_spelling(&layouts.in_packet, &positions);
    if layouts.is_declared(kept_elements, positions.len(), spelling.as_ref()) {
        return Ok(());
    }
    let layouts = Box::new(layouts);
    Err(match (spelling, layouts.has_padded_input()) {
        (Some(spelling), _) => FilterError::BitmapOutputMismatchesInputAxes {
            bitmap,
            spelling,
            layouts,
        },
        (None, true) => FilterError::BitmapNewAxesOverPaddedInput { bitmap, layouts },
        (None, false) => FilterError::BitmapOutputNeedsNewAxes { bitmap, layouts },
    })
}

impl FilterInput {
    /// Checks that the output preserves `Time` and describes the selected positions.
    fn is_declared(&self, kept_elements: usize, kept_per_packet: usize, spelling: Option<&Mapping>) -> bool {
        let output = self.output().normalize();
        let kept = output.clone().resize(kept_elements).normalize();
        // Only `#` may complete the last packet after the selected positions.
        if output != kept.clone().padding(output.size(), PaddingKind::Top).normalize() {
            return false;
        }
        let Ok((from_time, from_packet)) = kept.split_at(kept_per_packet) else {
            return false;
        };
        // `split_at` leaves composite pieces such as `(R, F = 4) / 4`, which normalizing unfolds.
        let (from_time, from_packet) = (from_time.normalize(), from_packet.normalize());
        if from_time != self.in_time.normalize() {
            return false;
        }

        // The output may spell the selected positions with the input axes, as `spelling` does.
        // TODO: Let the output forget a selected padding's kind as `#`, and declare the last packet's
        // tail `#{0}`, which the device fills with zeros (Arena 329087).
        if let Some(spelling) = spelling
            && from_packet == spelling.clone().normalize()
        {
            return true;
        }

        // Otherwise the output must name the selected positions with new axes.
        self.is_new_axes(&from_packet)
    }

    /// Whether the input `Time` × `Packet` holds padding anywhere, which rules out new axes whichever
    /// positions the bitmap keeps.
    fn has_padded_input(&self) -> bool {
        self.input().normalize().has_padding()
    }

    /// Checks that new axes give each selected position a distinct index.
    fn is_new_axes(&self, from_packet: &Mapping) -> bool {
        // A resize can hide a broadcast from the axis terms.
        let mut cells = from_packet
            .iter(&from_packet.axes(), &Index::new(), true)
            .collect::<Vec<_>>();
        cells.sort_unstable();
        cells.dedup();
        self.names_only_new_axes(from_packet)
            && from_packet.has_no_padding()
            && !self.has_padded_input()
            && cells.len() == from_packet.size()
    }

    pub(super) fn mismatches_input_axes_message(&self, bitmap: CollectBitmap, spelling: &Mapping) -> String {
        let Self {
            in_time,
            in_packet,
            out_time,
            out_packet,
        } = self;
        // Normalize before padding so the diagnostic shows the padding at the end.
        let kept = in_time.clone().pair(spelling.clone()).normalize();
        let volume = kept.size().next_multiple_of(ONE_FLIT_ELEMENTS);
        let output = kept.padding(volume, PaddingKind::Top);
        let new_axes = match self.has_padded_input() {
            // New axes need an input without padding.
            true => String::new(),
            false => format!(
                ", or Time {in_time} followed by new axes of {} elements",
                spelling.size()
            ),
        };
        format!(
            "the bitmap {:#010b} keeps {} of every packet {in_packet}, so the output is {output} split at \
             {ONE_FLIT_ELEMENTS}{new_axes}; got Time2 = {out_time}, Packet2 = {out_packet}",
            bitmap.bits(),
            spelling.normalize(),
        )
    }

    pub(super) fn new_axes_over_padded_input_message(&self, bitmap: CollectBitmap) -> String {
        format!(
            "the bitmap {:#010b} keeps positions that no input axes spell in {}, and new axes need the whole input \
             Time x Packet to hold no padding, whichever positions the bitmap keeps",
            bitmap.bits(),
            self.input_layout(),
        )
    }

    pub(super) fn needs_new_axes_message(&self, bitmap: CollectBitmap) -> String {
        let Self {
            in_time,
            in_packet,
            out_time,
            out_packet,
        } = self;
        let kept = bitmap.positions().count();
        format!(
            "the bitmap {:#010b} keeps {kept} positions of every packet that no input axes spell in {in_packet}, \
             so the output is Time {in_time} followed by new axes of {kept} elements, padded to whole packets and \
             split at {ONE_FLIT_ELEMENTS}; got Time2 = {out_time}, Packet2 = {out_packet}",
            bitmap.bits(),
        )
    }
}

/// Finds an input-axis mapping for sorted, distinct packet positions when possible.
fn input_axes_spelling(packet: &Mapping, positions: &[usize]) -> Option<Mapping> {
    let run_len = (positions.iter().enumerate())
        .take_while(|&(index, &position)| position == index)
        .count();
    if run_len == positions.len() {
        return Some(packet.clone().resize(run_len));
    }
    if run_len == 0 {
        return None;
    }
    // The second run's start gives the block size. For [0, 1, 4, 5], each block has
    // four positions and keeps its first two, giving `[F / 4, F % 4 = 2]` for axis `F`.
    let block_size = positions[run_len];
    if !packet.size().is_multiple_of(block_size) || !positions.len().is_multiple_of(run_len) {
        return None;
    }
    let blocks = positions.chunks_exact(run_len).map(|run| {
        let starts_a_block = run[0].is_multiple_of(block_size);
        let is_contiguous = (run.iter().enumerate()).all(|(index, &position)| position == run[0] + index);
        (starts_a_block && is_contiguous).then_some(run[0] / block_size)
    });
    let outer = input_axes_spelling(&packet.clone().stride(block_size), &blocks.collect::<Option<Vec<_>>>()?)?;
    Some(outer.pair(packet.clone().modulo(block_size).resize(run_len)))
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    fn bitmap(bits: u8) -> CollectBitmap {
        CollectBitmap::from(bits)
    }

    axes![T = 2, P = 8, N = 3, K = 16, R = 8, F = 8, S7 = 7, U2 = 2];

    macro_rules! by_bitmap {
        ($bits:expr, $in_time:ty, $in_packet:ty => $out_time:ty, $out_packet:ty) => {
            config_filter_by_bitmap(
                FilterInput {
                    in_time: <$in_time>::to_value(),
                    in_packet: <$in_packet>::to_value(),
                    out_time: <$out_time>::to_value(),
                    out_packet: <$out_packet>::to_value(),
                },
                bitmap($bits),
            )
        };
    }

    #[test]
    fn a_bitmap_drops_every_kind_of_padding() {
        assert_eq!(by_bitmap!(0b1, m![N], m![1 # 8] => m![1], m![N # 8]), Ok(()));
        assert_eq!(by_bitmap!(0b1, m![N], m![1 #{0} 8] => m![1], m![N # 8]), Ok(()));
        assert_eq!(by_bitmap!(0b1, m![N], m![1 #{!} 8] => m![1], m![N # 8]), Ok(()));
    }

    #[test]
    fn a_bitmap_output_spells_the_kept_positions_with_the_input_axes() {
        axes![X = 4, Y = 2, D = 4, K4 = 4, H = 2];
        assert_eq!(
            by_bitmap!(0b0000_1111, m![R], m![F] => m![[R, F = 4] / 8], m![[R, F = 4] % 8]),
            Ok(())
        );
        assert_eq!(
            by_bitmap!(0b0101_0101, m![R], m![F] => m![[R, F / 2] / 8], m![[R, F / 2] % 8]),
            Ok(())
        );
        assert!(matches!(
            by_bitmap!(0b0000_1111, m![R], m![F] => m![[R, F % 4] / 8], m![[R, F % 4] % 8]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b0011_0011, m![R], m![F] => m![[R, F / 4, F % 2] / 8], m![[R, F / 4, F % 2] % 8]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        assert_eq!(
            by_bitmap!(0b0011_0011, m![R], m![F] => m![[R, F / 4, F % 4 = 2] / 8], m![[R, F / 4, F % 4 = 2] % 8]),
            Ok(())
        );
        // Positions 0, 2, 4 and 6 of `[X, Y]` are `Y = 0`, leaving `X`.
        assert_eq!(
            by_bitmap!(0b0101_0101, m![R], m![X, Y] => m![[R, X] / 8], m![[R, X] % 8]),
            Ok(())
        );
        // Positions 0 and 4 from the two packets recover all of `D`.
        assert_eq!(
            by_bitmap!(0b0001_0001, m![D / 2], m![D % 2, 1 # 4] => m![1], m![D # 8]),
            Ok(())
        );
        // The repacked packets cross row boundaries when each row contributes seven elements.
        assert_eq!(
            by_bitmap!(0b0111_1111, m![R], m![S7 # 8] => m![[R, S7] / 8], m![[R, S7] % 8]),
            Ok(())
        );
        assert_eq!(
            by_bitmap!(0b0000_1111, m![K / 4], m![K % 4 # 8] => m![K / 8], m![K % 8]),
            Ok(())
        );
        let error = by_bitmap!(0b0101_0101, m![R], m![X, Y] => m![[X, R] / 8], m![[X, R] % 8]).unwrap_err();
        assert!(matches!(error, FilterError::BitmapOutputMismatchesInputAxes { .. }));
        let answer = "the output is (R, X) split at 8";
        assert!(error.to_string().contains(answer), "{error}");
        assert!(matches!(
            by_bitmap!(0b0000_1111, m![K / 4], m![K % 4 # 8] => m![H], m![F]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
    }

    #[test]
    fn a_bitmap_output_pads_only_after_the_kept_elements() {
        axes![T3 = 3];
        // Twelve selected elements fit in two packets.
        assert_eq!(
            by_bitmap!(0b1111, m![T3], m![F] => m![[T3, F = 4] # 16 / 8], m![[T3, F = 4] # 16 % 8]),
            Ok(())
        );
        assert!(matches!(
            by_bitmap!(0b1111, m![T3], m![F] => m![T3 # 4], m![F]),
            Err(FilterError::OutputVolume { kept_elements: 12, .. })
        ));
    }

    #[test]
    fn a_bitmap_output_names_the_kept_positions_with_new_axes() {
        axes![G = 3, U = 2, V = 2, E4 = 4];
        assert_eq!(
            by_bitmap!(0b1000_1001, m![R], m![F] => m![[R, G] / 8], m![[R, G] % 8]),
            Ok(())
        );
        assert_eq!(
            by_bitmap!(0b1100_0011, m![R], m![F] => m![[R, U, V] / 8], m![[R, U, V] % 8]),
            Ok(())
        );
        // Selecting one position leaves only the input Time axis.
        assert_eq!(by_bitmap!(0b1000, m![R], m![F] => m![1], m![R]), Ok(()));
        assert_eq!(
            by_bitmap!(0b0101_0101, m![R], m![F] => m![[R, E4] / 8], m![[R, E4] % 8]),
            Ok(())
        );
        // `F = 3` names positions 0, 1, and 2.
        assert!(matches!(
            by_bitmap!(0b1000_1001, m![R], m![F] => m![[R, F = 3] / 8], m![[R, F = 3] % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b1010_1010, m![R], m![F] => m![[R, F / 2] / 8], m![[R, F / 2] % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b0110_0011, m![R], m![F] => m![[R, F = 4] / 8], m![[R, F = 4] % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b1000_1001, m![R], m![F] => m![[G, R] / 8], m![[G, R] % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b1000_1001, m![R], m![F] => m![[R, U] # 24 / 8], m![[R, U] # 24 % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
        // `3` broadcasts a single index across all three selected positions.
        assert!(matches!(
            by_bitmap!(0b1000_1001, m![R], m![F] => m![[R, 3] / 8], m![[R, 3] % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
    }

    #[test]
    fn a_kept_padding_position_stays_padding() {
        axes![X = 2, Y = 2, E = 11];
        // The masks retain neither padding group, then positions 4 and 6, then positions 1 and 3.
        assert_eq!(
            by_bitmap!(0b0000_0101, m![Y], m![1 # 2, X, 1 # 2] => m![1], m![[Y, X] # 8]),
            Ok(())
        );
        assert_eq!(
            by_bitmap!(0b0101_0101, m![Y], m![1 # 2, X, 1 # 2] => m![1], m![Y, 1 # 2, X]),
            Ok(())
        );
        assert_eq!(
            by_bitmap!(0b0000_1111, m![Y], m![1 # 2, X, 1 # 2] => m![1], m![Y, X, 1 # 2]),
            Ok(())
        );
        assert_eq!(
            by_bitmap!(0b0011_0011, m![Y], m![X, 1 # 4] => m![1], m![Y, X, 1 # 2]),
            Ok(())
        );
        // Input Time padding survives repacking.
        assert_eq!(by_bitmap!(0b1, m![T # 4], m![F] => m![1], m![T # 8]), Ok(()));
        assert_eq!(
            by_bitmap!(0b1111_1111, m![E # 16 / 8], m![E # 16 % 8] => m![E # 16 / 8], m![E # 16 % 8]),
            Ok(())
        );
        assert!(matches!(
            by_bitmap!(0b0101_0101, m![Y], m![1 # 2, X, 1 # 2] => m![1], m![Y, 1 #{0} 2, X]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b0101_0101, m![Y], m![1 # 2, X, 1 # 2] => m![1], m![Y, 1 #{!} 2, X]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        assert_eq!(
            by_bitmap!(0b0101_0101, m![Y], m![1 #{0} 2, X, 1 # 2] => m![1], m![Y, 1 #{0} 2, X]),
            Ok(())
        );
        assert!(matches!(
            by_bitmap!(0b0101_0101, m![Y], m![1 #{0} 2, X, 1 # 2] => m![1], m![Y, 1 # 2, X]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b0101_0101, m![Y], m![1 # 2, X, 1 # 2] => m![1], m![Y, X, 1 # 2]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
    }

    #[test]
    fn new_axes_rename_only_a_padding_free_input() {
        axes![E = 11, P4 = 4, G = 3, A2 = 2, C10 = 10];
        // These masks keep some padding, then only padding, at positions no input axes spell.
        assert!(matches!(
            by_bitmap!(0b0001_1001, m![R], m![P4 # 8] => m![[R, G] / 8], m![[R, G] % 8]),
            Err(FilterError::BitmapNewAxesOverPaddedInput { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b1010_0000, m![R], m![P4 # 8] => m![[R, U2] / 8], m![[R, U2] % 8]),
            Err(FilterError::BitmapNewAxesOverPaddedInput { .. })
        ));
        // Position 7 holds an element in the first packet and padding in the second.
        assert!(matches!(
            by_bitmap!(0b1000_0001, m![E # 16 / 8], m![E # 16 % 8] => m![1], m![[E # 16 / 8, U2] # 8]),
            Err(FilterError::BitmapNewAxesOverPaddedInput { .. })
        ));
        // Positions 0 and 5 hold elements in all three packets, but the input holds padding.
        assert!(matches!(
            by_bitmap!(
                0b0010_0001, m![[A2, C10 # 12] / 8], m![[A2, C10 # 12] % 8] => m![1], m![[[A2, C10 # 12] / 8, U2] # 8]
            ),
            Err(FilterError::BitmapNewAxesOverPaddedInput { .. })
        ));
        // The input axes spell kept padding, so only that spelling is accepted.
        assert!(matches!(
            by_bitmap!(0b11, m![R], m![1 # 8] => m![[R, U2] / 8], m![[R, U2] % 8]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
    }

    #[test]
    fn new_axes_hold_no_padding() {
        axes![G = 3];
        assert_eq!(
            by_bitmap!(0b0001_1001, m![R], m![F] => m![[R, G] / 8], m![[R, G] % 8]),
            Ok(())
        );
        assert!(matches!(
            by_bitmap!(0b0001_1001, m![R], m![F] => m![[R, U2 # 3] / 8], m![[R, U2 # 3] % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
    }

    #[test]
    fn a_bitmap_needs_a_kept_position_and_whole_packets() {
        axes![P4 = 4];
        assert_eq!(
            by_bitmap!(0, m![T], m![F] => m![1], m![T # 8]),
            Err(FilterError::EmptyBitmap)
        );
        assert!(matches!(
            by_bitmap!(0b1, m![T], m![P4] => m![1], m![T # 8]),
            Err(FilterError::PacketSize { .. })
        ));
    }

    #[test]
    fn every_packet_of_an_input_axes_spelling_is_checked() {
        axes![E = 11, A2 = 2, C10 = 10];
        // Position 7 is padding only in the second packet.
        assert!(matches!(
            by_bitmap!(0b1111_1111, m![E # 16 / 8], m![E # 16 % 8] => m![E # 16 / 8], m![E #{0} 16 % 8]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        // `C10 = 8` describes the first packet. The second crosses into the next `A2` row.
        assert!(matches!(
            by_bitmap!(
                0b1111_1111, m![[A2, C10 # 12] / 8], m![[A2, C10 # 12] % 8] => m![[A2, C10 # 12] / 8], m![C10 = 8]
            ),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
    }

    #[test]
    fn a_bitmap_output_checks_each_of_its_conditions() {
        axes![X4 = 4, W = 2, T3 = 3, X = 2, Y = 2];
        // The first four cells match. The next four should be padding.
        assert!(matches!(
            by_bitmap!(0b1111, m![1], m![F] => m![1], m![F]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        // Broadcasting allows the cells to match despite mixing input and new axes.
        assert!(matches!(
            by_bitmap!(0b0000_1111, m![R], m![X4, 2] => m![[R, X4 = 2, W] / 8], m![[R, X4 = 2, W] % 8]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        // Positions 0, 2, and 6 skip block 2 of `F / 2`.
        assert!(matches!(
            by_bitmap!(0b0100_0101, m![R], m![F] => m![[R, F = 3] / 8], m![[R, F = 3] % 8]),
            Err(FilterError::BitmapOutputNeedsNewAxes { .. })
        ));
        assert_eq!(
            by_bitmap!(0b0101_0101, m![Y], m![1 #{!} 2, X, 1 # 2] => m![1], m![Y, 1 #{!} 2, X]),
            Ok(())
        );
        assert!(matches!(
            by_bitmap!(0b0101_0101, m![Y], m![1 #{!} 2, X, 1 # 2] => m![1], m![Y, 1 # 2, X]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b0101_0101, m![Y], m![1 #{!} 2, X, 1 # 2] => m![1], m![Y, 1 #{0} 2, X]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        assert!(matches!(
            by_bitmap!(0b0101_0101, m![Y], m![1 # 2, X, 1 # 2] => m![1], m![Y, 1 # 2, 1 # 2]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
        // Padding added to finish the last packet must use `#`.
        assert!(matches!(
            by_bitmap!(0b1111, m![T3], m![F] => m![[T3, F = 4] #{0} 16 / 8], m![[T3, F = 4] #{0} 16 % 8]),
            Err(FilterError::BitmapOutputMismatchesInputAxes { .. })
        ));
    }
}
