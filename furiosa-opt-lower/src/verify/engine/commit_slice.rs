//! Which slices a commit writes, and which stream slices may reach them.

use furiosa_mapping::{Cell, Ident, IndexExt, Mapping, MappingExt, PaddingKind, RSortedMap};

/// The `Slice` placements a commit reads from and writes to.
pub struct CommitSliceInput {
    pub stream: Mapping,
    pub destination: Mapping,
    pub packets: StreamPackets,
}

/// How many packets each slice of a committed stream carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPackets {
    /// Every slice carries the same packets.
    Fixed,
    /// A tag filter's survivors decide each slice's packets.
    TagFiltered,
}

/// A `Slice` pair no commit can take.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommitSliceError {
    #[error("the stream has {stream_slices} slices but the destination has {destination_slices}")]
    SliceCountMismatch {
        stream_slices: usize,
        destination_slices: usize,
    },
    #[error(
        "slice {slice_position} is undefined in the stream ({stream}), so the destination ({destination}) must leave it \
         undefined (`#`) or unwritten (`#{{!}}`)"
    )]
    UndefinedSlice {
        slice_position: usize,
        stream: Mapping,
        destination: Mapping,
    },
    #[error(
        "slice {slice_position} is undefined in the tag-filtered stream ({stream}), so its undefined values decide how \
         many packets it emits, and a slice that emits past its output stops the device; the destination \
         ({destination}) must skip it (`#{{!}}`)"
    )]
    UndefinedFilteredSlice {
        slice_position: usize,
        stream: Mapping,
        destination: Mapping,
    },
    #[error(
        "slice {slice_position} holds a value in the stream ({stream}), but the destination ({destination}) promises \
         zero there (`#{{0}}`): pad the destination with `#` to write the value or with `#{{!}}` to skip it"
    )]
    NonZeroSlice {
        slice_position: usize,
        stream: Mapping,
        destination: Mapping,
    },
    #[error(
        "slice {slice_position} has coordinates {stream_coordinates:?} in the stream ({stream}) but \
         {destination_coordinates:?} in the destination ({destination}): a commit writes each slice in place and \
         compares the coordinates axis by axis, by axis name, so the destination's Slice must be the stream's with \
         part of it narrowed and padded (`X = k # n` writes the \
         rest as don't-care, `X = k #{{!}} n` skips it, and `k = 1` keeps only `X = 0`)"
    )]
    MovedSlice {
        slice_position: usize,
        stream: Mapping,
        destination: Mapping,
        stream_coordinates: Box<RSortedMap<Ident, usize>>,
        destination_coordinates: Box<RSortedMap<Ident, usize>>,
    },
}

/// What one slice of a placement holds.
enum SliceContent {
    /// The slice's logical coordinates, per axis with coordinate 0 dropped.
    Value(RSortedMap<Ident, usize>),
    /// `Zero` padding: the slice holds the value zero.
    Zero,
    /// `Top` or `Bottom` padding: the slice holds no value.
    NoValue,
}

/// Rejects a slice-count mismatch and the slices no commit can write in place:
///
/// | stream \ destination | Top | Zero | Bottom | Index |
/// | --- | --- | --- | --- | --- |
/// | Top, Bottom | ok, `UndefinedFilteredSlice` if tag-filtered | `UndefinedSlice` | ok | `UndefinedSlice` |
/// | Zero | ok | ok | ok | ok |
/// | Index | ok | `NonZeroSlice` | ok | `MovedSlice` unless the coordinates agree |
///
/// Whether an accepted slice is written is the destination's alone: a `Bottom` slice is skipped
/// ([`commit_skipped_slices`]).
pub fn config_commit_slice(input: CommitSliceInput) -> Result<(), CommitSliceError> {
    let CommitSliceInput {
        stream,
        destination,
        packets,
    } = input;
    if stream.size() != destination.size() {
        return Err(CommitSliceError::SliceCountMismatch {
            stream_slices: stream.size(),
            destination_slices: destination.size(),
        });
    }
    for slice_position in 0..stream.size() {
        match (
            slice_content(&stream, slice_position),
            slice_content(&destination, slice_position),
        ) {
            // A slice with no value cannot reach one that promises a value.
            (SliceContent::NoValue, SliceContent::Value(_) | SliceContent::Zero) => {
                return Err(CommitSliceError::UndefinedSlice {
                    slice_position,
                    stream,
                    destination,
                });
            }
            // A written slice's undefined values could pass the guard and overflow (Arena 329503),
            // while a skipped one cannot (Arena 329504).
            (SliceContent::NoValue, SliceContent::NoValue)
                if packets == StreamPackets::TagFiltered && !is_skipped(&destination, slice_position) =>
            {
                return Err(CommitSliceError::UndefinedFilteredSlice {
                    slice_position,
                    stream,
                    destination,
                });
            }
            // The commit would write the stream's value where the destination promises zero.
            (SliceContent::Value(_), SliceContent::Zero) => {
                return Err(CommitSliceError::NonZeroSlice {
                    slice_position,
                    stream,
                    destination,
                });
            }
            // A commit writes each slice in place. Coordinates are per axis, so a tile's `B % 4`
            // gives the same Index as the base's `B`.
            (SliceContent::Value(stream_coordinates), SliceContent::Value(destination_coordinates))
                if stream_coordinates != destination_coordinates =>
            {
                return Err(CommitSliceError::MovedSlice {
                    slice_position,
                    stream,
                    destination,
                    stream_coordinates: Box::new(stream_coordinates),
                    destination_coordinates: Box::new(destination_coordinates),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// The positions of the destination's `Bottom` slices, ascending.
pub fn commit_skipped_slices(destination: &Mapping) -> Vec<usize> {
    (0..destination.size())
        .filter(|&position| is_skipped(destination, position))
        .collect()
}

/// Whether a commit skips slice `position` of `destination`.
fn is_skipped(destination: &Mapping, position: usize) -> bool {
    matches!(destination.index(position), Cell::Padding(PaddingKind::Bottom))
}

/// What slice `position` of `placement` holds. A cell whose coordinates overshoot reads as the
/// padding it lands on.
fn slice_content(placement: &Mapping, position: usize) -> SliceContent {
    let padding = match placement.index(position) {
        Cell::Index(index) => match index.finalize().into_result() {
            Ok(coordinates) => return SliceContent::Value(coordinates),
            Err(padding) => padding,
        },
        Cell::Padding(padding) => padding,
        Cell::OutOfBounds => unreachable!("a slice position below the placement's size is in bounds"),
    };
    match padding {
        PaddingKind::Zero => SliceContent::Zero,
        PaddingKind::Top | PaddingKind::Bottom => SliceContent::NoValue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_mapping::*;

    axes![A = 8, B = 8];

    fn commit_slice(stream: Mapping, destination: Mapping) -> Result<(), CommitSliceError> {
        config_commit_slice(CommitSliceInput {
            stream,
            destination,
            packets: StreamPackets::Fixed,
        })
    }

    fn commit_filtered_slice(stream: Mapping, destination: Mapping) -> Result<(), CommitSliceError> {
        config_commit_slice(CommitSliceInput {
            stream,
            destination,
            packets: StreamPackets::TagFiltered,
        })
    }

    #[test]
    fn an_undefined_slice_of_a_tag_filtered_stream_commits_only_to_a_skipped_slice() {
        let stream = <m![A, B % 4 # 8]>::to_value();
        assert_eq!(
            commit_filtered_slice(stream.clone(), <m![A, B % 4 #{!} 8]>::to_value()),
            Ok(())
        );
        assert!(matches!(
            commit_filtered_slice(stream, <m![A, B % 4 # 8]>::to_value()),
            Err(CommitSliceError::UndefinedFilteredSlice { slice_position: 4, .. })
        ));
        assert!(matches!(
            commit_filtered_slice(<m![A, B % 4 #{!} 8]>::to_value(), <m![A, B % 4 # 8]>::to_value()),
            Err(CommitSliceError::UndefinedFilteredSlice { slice_position: 4, .. })
        ));
    }

    #[test]
    fn a_commit_skips_only_the_bottom_slices() {
        assert_eq!(
            commit_skipped_slices(&<m![A, B % 4 #{!} 8]>::to_value()),
            (0..64).filter(|slice| slice % 8 >= 4).collect::<Vec<_>>()
        );
        assert_eq!(
            commit_skipped_slices(&<m![A, B % 4 # 8]>::to_value()),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn an_undefined_slice_commits_only_to_a_padding_slice() {
        let stream = <m![A, B % 4 # 8]>::to_value();
        assert_eq!(commit_slice(stream.clone(), <m![A, B % 4 #{!} 8]>::to_value()), Ok(()));
        assert_eq!(commit_slice(stream.clone(), stream.clone()), Ok(()));
        assert!(matches!(
            commit_slice(stream.clone(), <m![A, B]>::to_value()),
            Err(CommitSliceError::UndefinedSlice { slice_position: 4, .. })
        ));
        // `Zero` padding promises zeros, which an undefined slice would break.
        assert!(matches!(
            commit_slice(stream, <m![A, B % 4 #{0} 8]>::to_value()),
            Err(CommitSliceError::UndefinedSlice { slice_position: 4, .. })
        ));
    }

    #[test]
    fn a_bottom_slice_of_the_stream_is_undefined() {
        assert!(matches!(
            commit_slice(<m![A, B % 4 #{!} 8]>::to_value(), <m![A, B]>::to_value()),
            Err(CommitSliceError::UndefinedSlice { slice_position: 4, .. })
        ));
    }

    #[test]
    fn a_zero_slice_of_the_stream_holds_a_value() {
        assert_eq!(
            commit_slice(<m![A, B % 4 #{0} 8]>::to_value(), <m![A, B]>::to_value()),
            Ok(())
        );
    }

    #[test]
    fn only_a_zero_slice_commits_to_a_zero_slice() {
        let destination = <m![A, B % 4 #{0} 8]>::to_value();
        assert_eq!(commit_slice(destination.clone(), destination.clone()), Ok(()));
        assert!(matches!(
            commit_slice(<m![A, B]>::to_value(), destination),
            Err(CommitSliceError::NonZeroSlice { slice_position: 4, .. })
        ));
    }

    #[test]
    fn a_valid_slice_may_go_where_the_destination_skips_it() {
        assert_eq!(
            commit_slice(<m![A, B]>::to_value(), <m![A, B % 4 #{!} 8]>::to_value()),
            Ok(())
        );
    }

    #[test]
    fn the_stream_and_the_destination_have_as_many_slices() {
        assert!(matches!(
            commit_slice(<m![A, B % 4 # 8]>::to_value(), <m![A]>::to_value()),
            Err(CommitSliceError::SliceCountMismatch {
                stream_slices: 64,
                destination_slices: 8
            })
        ));
    }

    #[test]
    fn a_commit_cannot_move_a_slice() {
        // Slice 1 is `A = 0, B = 1` in the stream and `A = 1, B = 0` in the destination.
        assert!(matches!(
            commit_slice(<m![A, B]>::to_value(), <m![B, A]>::to_value()),
            Err(CommitSliceError::MovedSlice { slice_position: 1, .. })
        ));
    }

    #[test]
    fn a_commit_cannot_reorder_the_pieces_of_one_axis() {
        // Slice 1 is `B = 1` in the stream and `B = 2` in the destination.
        assert!(matches!(
            commit_slice(<m![A, B]>::to_value(), <m![A, B % 2, B / 2]>::to_value()),
            Err(CommitSliceError::MovedSlice { slice_position: 1, .. })
        ));
    }

    #[test]
    fn a_destination_narrows_stream_axes_to_a_prefix() {
        axes![A2 = 2, B4 = 4, C8 = 8];
        let stream = <m![A2, B4, C8]>::to_value();
        for destination in [
            <m![A2, B4, 1 #{!} 8]>::to_value(),
            <m![A2, B4, 1 # 8]>::to_value(),
            <m![A2, B4, C8 = 4 #{!} 8]>::to_value(),
        ] {
            assert_eq!(
                commit_slice(stream.clone(), destination.clone()),
                Ok(()),
                "{destination}"
            );
        }
        // Slice 1 is `C8 = 1` in the stream but `B4 = 1` in the destination.
        assert!(matches!(
            commit_slice(stream, <m![A2, C8, B4]>::to_value()),
            Err(CommitSliceError::MovedSlice { slice_position: 1, .. })
        ));
    }
}
