//! Commit Engine: Tensor Unit stream to DM.
//!
//! Drains a post-Commit-Adapter `TuTensor` (from `commit_trim` onwards) into a
//! [`DmTensor`] (or an existing mutable view). `verify_commit` synthesizes the
//! write sequencer through `config_commit`, the dual of the fetch read.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use furiosa_opt_lower::{
    CommitInput, CommitSliceInput, StreamPackets, commit_skipped_slices, config_commit, config_commit_slice,
};

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::CanApplyCommit;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::memory::{DmTensor, DmTensorViewMut};
use crate::tensor::tu::{TagFiltered, TuTensor};

// ANCHOR: commit_impl
// `D: MaterializableScalar`: a commit writes values data memory can hold.
impl<
    'l,
    const T: Tu,
    P: CanApplyCommit,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Commits to data memory.
    #[primitive(TuTensor::commit)]
    pub fn commit<Element: M>(self) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        let mut output = DmTensor::from_parts(Tensor::zeroed());
        self.commit_view(output.view_mut());
        output
    }

    /// Commits to a mutable DM view, skipping its `Bottom` slices.
    /// Panics when a slice would move or an undefined stream slice would reach a defined one.
    #[primitive(TuTensor::commit_view)]
    pub fn commit_view<DstSlice: M, Element: M>(
        self,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, DstSlice, Element, B>,
    ) {
        constraints::assert_slice_preserved::<Slice, DstSlice>();
        verify_commit_view_layout(
            CommitSliceInput {
                stream: Slice::to_value(),
                destination: DstSlice::to_value(),
                packets: StreamPackets::Fixed,
            },
            CommitInput {
                in_time: Time::to_value(),
                in_packet: Packet::to_value(),
                element: Element::to_value(),
                element_bits: D::BITS,
            },
        );
        // A commit writes each slice in place, so the stream's slice `p` is read as the
        // destination's slice `p`; the stream may hold axes the destination's `Slice` narrows away.
        // SAFETY: the relabel keeps every slice at its position, and `verify_commit_view_layout` has
        // checked that both name the same coordinates wherever the destination holds a value.
        let in_place = unsafe {
            self.inner
                .reshape::<m![{ Chip }, { Cluster }, { DstSlice }, { Time }, { Packet }]>()
        };
        dst.inner.transpose(in_place.view(), false);
    }
}
// ANCHOR_END: commit_impl

// A tag filter's stream reaches data memory through this `commit_view` alone.
impl<
    'l,
    const T: Tu,
    P: CanApplyCommit,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B, TagFiltered>
{
    /// Writes the packets that arrive into `dst`, preserving every destination value they do not
    /// reach. A slice keeps its leading `Time` packets and drops the rest, as the device does.
    #[primitive(TuTensor::commit_view)]
    pub fn commit_view<DstSlice: M, Element: M>(
        self,
        mut dst: DmTensorViewMut<'l, D, Chip, Cluster, DstSlice, Element, B>,
    ) {
        constraints::assert_slice_preserved::<Slice, DstSlice>();
        verify_commit_view_layout(
            CommitSliceInput {
                stream: Slice::to_value(),
                destination: DstSlice::to_value(),
                packets: StreamPackets::TagFiltered,
            },
            CommitInput {
                in_time: Time::to_value(),
                in_packet: Packet::to_value(),
                element: Element::to_value(),
                element_bits: D::BITS,
            },
        );
        // 1. Count the leading elements of each slice that the commit writes: the packets its
        //    filter emitted. A commit skips the destination's `Bottom` slices, so none of their
        //    elements count. Every chip and cluster repeats the destination's `Slice`.
        let TagFiltered(filtered) = &self.time_extent;
        let skipped = commit_skipped_slices(&DstSlice::to_value());
        let arrived = (filtered.written_packets(|slice| !skipped.contains(&(slice % DstSlice::SIZE))))
            .unwrap_or_else(|error| panic!("{error}"))
            .into_iter()
            .map(|packets| packets.0 * Packet::SIZE)
            .collect::<Vec<_>>();
        // 2. Read what the destination holds now, in the stream's layout. A commit writes each slice
        //    in place, so slice `p` of one is slice `p` of the other.
        let previous = dst
            .inner
            .read()
            .transpose::<m![{ Chip }, { Cluster }, { DstSlice }, { Time }, { Packet }]>(false)
            .into_vec();
        // 3. Past its counted elements a slice of the stream holds padding. Put the destination's
        //    own values there.
        let mut stream = self.inner.into_vec();
        restore_unwritten(&mut stream, &previous, &arrived);
        // 4. Write the whole stream. Only the counted elements differ from the destination.
        let stream: Tensor<D, m![{ Chip }, { Cluster }, { DstSlice }, { Time }, { Packet }], B> =
            Tensor::from_vec(stream);
        dst.inner.transpose(stream.view(), false);
    }
}

/// Checks the slices and packet layout shared by both `commit_view` entry points.
fn verify_commit_view_layout(slices: CommitSliceInput, commit: CommitInput) {
    config_commit_slice(slices).unwrap_or_else(|error| panic!("{error}"));
    verify_commit(commit);
}

/// Validates the Commit engine and synthesizes the write config via [`config_commit`], panicking
/// with the commit error on failure. The resolved descriptors are not consumed yet, so they're
/// explicitly discarded. The dual of `verify_fetch`: it writes the time-ordered `(InTime, InPacket)`
/// stream into the DM `Element` layout.
pub(crate) fn verify_commit(input: CommitInput) {
    let _ = config_commit(input).unwrap_or_else(|e| panic!("{e}"));
}

/// Restores each slice of `stream` past its `arrived` elements to the same elements of `previous`.
fn restore_unwritten<V: Clone>(stream: &mut [V], previous: &[V], arrived: &[usize]) {
    let slice_elements = stream.len() / arrived.len();
    for (slice, arrived) in arrived.iter().enumerate() {
        let unwritten = slice * slice_elements + arrived..(slice + 1) * slice_elements;
        stream[unwritten.clone()].clone_from_slice(&previous[unwritten]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_lower::CommitError;

    /// Each accepts a commit across an orthogonal phenomenon; the matcher's write synthesis is
    /// exhaustively tested in `furiosa-mapping-impl`.
    mod commit_valid {
        use super::*;

        axes![N = 8, A = 4, B = 3, C = 4];

        /// Full trim, then a clean multi-axis commit.
        #[test]
        fn full_trim_then_commit() {
            verify_commit(CommitInput {
                in_time: <m![A, B, C]>::to_value(),
                in_packet: <m![N]>::to_value(),
                element: <m![A, B, C, N]>::to_value(),
                element_bits: <i8 as crate::scalar::Scalar>::BITS,
            });
        }

        /// Packet keeps trailing flit padding (partial trim).
        #[test]
        fn partial_trim_then_commit() {
            verify_commit(CommitInput {
                in_time: <m![A]>::to_value(),
                in_packet: <m![N # 16]>::to_value(),
                element: <m![A, N # 16]>::to_value(),
                element_bits: <i8 as crate::scalar::Scalar>::BITS,
            });
        }

        /// Time axes are transposed across the commit.
        #[test]
        fn time_transpose() {
            verify_commit(CommitInput {
                in_time: <m![A # 32, B]>::to_value(),
                in_packet: <m![N]>::to_value(),
                element: <m![B, A # 32, N]>::to_value(),
                element_bits: <i8 as crate::scalar::Scalar>::BITS,
            });
        }

        /// The stream emits the live block four times (`1#2 ⋈ 1#2`); overlapping
        /// dummy writes collapse onto the DM's smaller dummy region — realizable
        /// even though an injective placement could not fit.
        #[test]
        fn interleaved_time_padding_overlaps_into_dm_padding() {
            verify_commit(CommitInput {
                in_time: <m![1 # 2, A, 1 # 2]>::to_value(),
                in_packet: <m![N]>::to_value(),
                element: <m![A, 1 # 3, N]>::to_value(),
                element_bits: <i8 as crate::scalar::Scalar>::BITS,
            });
        }
    }
    /// One case per rejection path, checking the typed commit error directly.
    mod commit_invalid {
        use super::*;
        use furiosa_mapping::M as _;

        axes![M = 4, N = 8, X = 8, Y = 4, A = 4, B = 3];

        /// Packet byte size is not a legal flit trim (3 bytes).
        #[test]
        fn illegal_packet_size() {
            assert_eq!(
                config_commit(CommitInput {
                    in_time: <m![M]>::to_value(),
                    in_packet: <m![B]>::to_value(),
                    element: <m![M, B]>::to_value(),
                    element_bits: <i8 as Scalar>::BITS,
                }),
                Err(CommitError::IllegalPacketBytes { bytes: B::SIZE })
            );
        }

        /// Element's innermost chunk is not the input packet.
        #[test]
        fn packet_mismatch() {
            assert!(matches!(
                config_commit(CommitInput {
                    in_time: <m![M]>::to_value(),
                    in_packet: <m![N # 32]>::to_value(),
                    element: <m![M, X]>::to_value(),
                    element_bits: <i8 as Scalar>::BITS,
                }),
                Err(CommitError::PacketNotInnermost { .. })
            ));
        }

        /// Live Time terms do not correspond to the Element outer's terms.
        #[test]
        fn time_term_mismatch() {
            assert!(matches!(
                config_commit(CommitInput {
                    in_time: <m![M]>::to_value(),
                    in_packet: <m![N # 32]>::to_value(),
                    element: <m![Y, N # 32]>::to_value(),
                    element_bits: <i8 as Scalar>::BITS,
                }),
                Err(CommitError::Unwritable { .. })
            ));
        }

        /// The stream's padding over-emits beyond the DM's dummy capacity, so the matcher cannot match
        /// the over-emitting segment.
        #[test]
        fn no_affine_plan() {
            assert!(matches!(
                config_commit(CommitInput {
                    in_time: <m![A, 1 # 5]>::to_value(),
                    in_packet: <m![N]>::to_value(),
                    element: <m![A, 1 # 2, N]>::to_value(),
                    element_bits: <i8 as Scalar>::BITS,
                }),
                Err(CommitError::Unwritable { .. })
            ));
        }
    }
}
