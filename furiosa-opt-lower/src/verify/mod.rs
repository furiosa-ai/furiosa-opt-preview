//! Engine-constraint verification: the user-visible DSL contract for each Tensor-Unit engine.
//!
//! Unlike the lowering algorithms in the crate root (thin FFI wrappers over the hidden
//! `furiosa-opt-lower-impl`), these are plain, published functions. They state *what* mapping a stage
//! must produce, in pure `furiosa_mapping` terms. Both the frontend (`furiosa-opt-std`, whose
//! type-level `verify_*` delegate here) and the IR translator (`npu-visa-translate`) call them, so a
//! kernel is checked identically whether it runs under an interpreting backend or is compiled.
//!
//! Each check returns a typed, per-engine error enum carrying the offending mappings; both callers
//! render it (`panic!` in the frontend, a spanned diagnostic in the translator).
//!
//! When a check needs a hidden algorithm (e.g. exact division), it calls the FFI wrapper in the crate
//! root rather than reimplementing it, keeping the algorithm private while the constraint stays visible.
//!
//! The submodule layout follows `furiosa-opt-std/src/engine`: each verification lives under the matching
//! engine subtree. The vector engine's five checks are consolidated into one `vector/tensor.rs` rather
//! than split as they are in the frontend. `slice` sits outside `engine` since it verifies DMA
//! redistribution (`furiosa-opt-std/src/tensor/memory`), not a Tensor-Unit engine.

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use furiosa_mapping::{Mapping, MappingExt};

use crate::{DivideError, DivideInput, DivideTerm};

pub mod engine;
mod gather;
mod indirect;
mod scatter;
mod slice;

pub use engine::{
    CastError, CastInput, CastKind, CollectError, CollectInput, CommitCastError, CommitCastInput, CommitCastKind,
    CommitTrimError, CommitTrimInput, ContractLaneError, ContractLaneInput, ContractPacketError, ContractPacketInput,
    ContractTimeError, ContractTimeInput, FetchContext, FetchDimensionsInput, FetchLiftDimension, FetchLiftError,
    FetchLiftInput, FetchVolumeError, LaneMode, ReduceLabelInput, StreamAdapterError, StreamAdapterInput, ToTrfError,
    ToTrfInput, ToVrfError, ToVrfInput, UnreadableCause, VectorError, VectorIntraSliceUnzipInput,
    VectorNarrowSplitInput, VectorNarrowTrimInput, VectorWidenConcatInput, VectorWidenPadInput, VrfOperandError,
    VrfOperandInput, config_cast, config_collect, config_commit_cast, config_commit_trim, config_contract_lane,
    config_contract_packet, config_contract_time, config_fetch_dimensions, config_fetch_lift, config_fetch_volume,
    config_reduce_label, config_stream_adapter, config_to_trf, config_to_vrf, config_vector_intra_slice_unzip,
    config_vector_narrow_split, config_vector_narrow_trim, config_vector_widen_concat, config_vector_widen_pad,
    config_vrf_operand,
};
pub use gather::{
    GatherMappingError, GatherPayload, GatherValidationError, first_replica, gather_payload, validate_gather_output,
    validate_gather_payload,
};
pub use indirect::{
    DmPlacement, IndexChipError, IndexMapping, IndexPaddingError, SpmIndirectIndexError, SpmPlacement,
    validate_index_chip, validate_spm_indirect_index,
};
pub use scatter::{
    ScatterPayload, ScatterValidationError, scatter_payload, validate_scatter_payload, validate_scatter_updates,
};
pub use slice::*;

/// Why an element count cannot be represented as an exact byte count.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ElementSizeError {
    /// The element sequence ends inside a byte.
    #[error("{elements} elements x {element_bits} bits is not byte-aligned")]
    NotByteAligned { elements: usize, element_bits: usize },
    /// The byte sequence ends inside an element.
    #[error("{bytes} bytes does not contain a whole number of {element_bits}-bit elements")]
    NotElementAligned { bytes: usize, element_bits: usize },
}

/// Packet endpoint checked against a one-flit size constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketSide {
    Input,
    Output,
}

impl Display for PacketSide {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "input",
            Self::Output => "output",
        })
    }
}

/// A packet whose byte size is not exactly one flit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{side} packet must be exactly {FLIT_BYTES} bytes (one flit): {elements} elements = {bytes} bytes")]
pub struct OneFlitPacketError {
    pub side: PacketSide,
    pub elements: usize,
    pub bytes: usize,
}

/// Bits in a byte.
pub const BITS_PER_BYTE: usize = 8;
/// Size of a single flit in bytes; the switching network moves data in flit-sized units, and the
/// fetch network carries one flit per go.
pub const FLIT_BYTES: usize = 32;
/// Vector register file capacity in bytes, per slice. One `to_vrf` operand must fit this.
pub const VRF_BYTES: usize = 8 * 1024;
/// Scratchpad memory capacity in bytes, per fused PE. One `SpmTensor`'s `Element` must fit this.
pub const SPM_BYTES_PER_PE: usize = 4 * 1024;
/// Vector register-file cache capacity in bytes. One interleaved unzip group must fit this.
pub const VRF_CACHE_BYTES: usize = 1024;
/// Vector-engine element width in bits: the engine computes on 32-bit lanes only.
pub(crate) const VE_ELEMENT_BITS: usize = 32;
/// Bytes in one SRAM access word. A fetch reads whole words unless a cast widens it back to them.
pub const SRAM_ACCESS_BYTES: usize = 8;
/// Required address alignment for a DMA write into DM.
pub const DM_WRITE_ALIGN_BYTES: usize = 8;
/// Columns of the temporal accumulator (the packet reducer's output-width bound).
pub const TEMPORAL_ACCUMULATOR_COLS: usize = 32;
/// Elements of the lane-folder / packet-reducer output packet (one flit of `i32`/`f32`).
pub(crate) const CONTRACT_LANE_OUT_PACKET_ELEMENTS: usize = 8;
/// Elements in a one-flit vector packet (8 x 32-bit lanes), which is `Way8`'s ALU access width.
pub(crate) const ONE_FLIT_ELEMENTS: usize = 8;
/// Elements in a half-flit vector packet (the front-4 lanes, the partial-reduction width, and
/// `Way4`'s ALU access width).
pub(crate) const HALF_FLIT_ELEMENTS: usize = 4;

/// Returns the exact byte size of an element sequence.
pub(crate) fn size_in_bytes(element_bits: usize, elements: usize) -> Result<usize, ElementSizeError> {
    // An element width comes from a `Scalar`, and a mapping's extent is nowhere near the cells this
    // would need to overflow.
    let bits = elements
        .checked_mul(element_bits)
        .expect("an element count times its width stays inside usize");
    if !bits.is_multiple_of(BITS_PER_BYTE) {
        return Err(ElementSizeError::NotByteAligned { elements, element_bits });
    }
    Ok(bits / BITS_PER_BYTE)
}

pub(crate) fn require_one_flit(side: PacketSide, elements: usize, bytes: usize) -> Result<(), OneFlitPacketError> {
    if bytes != FLIT_BYTES {
        return Err(OneFlitPacketError { side, elements, bytes });
    }
    Ok(())
}

/// Returns the exact element count contained in a byte sequence.
pub(crate) fn length_from_bytes(element_bits: usize, bytes: usize) -> Result<usize, ElementSizeError> {
    let bits = bytes
        .checked_mul(BITS_PER_BYTE)
        .expect("a byte count times eight stays inside usize");
    if !bits.is_multiple_of(element_bits) {
        return Err(ElementSizeError::NotElementAligned { bytes, element_bits });
    }
    Ok(bits / element_bits)
}

/// `a` rounded up to a multiple of `b`.
pub(crate) fn align_up(a: usize, b: usize) -> usize {
    a.div_ceil(b) * b
}

/// The hardware supports 1, 2, 4, or 8 lanes.
pub(crate) fn is_valid_lane_size(size: usize) -> bool {
    matches!(size, 1 | 2 | 4 | 8)
}

/// Padded extent of each axis at its cumulative stride; the contraction verifies key into this by a
/// division term's `dividend_stride`.
pub(crate) fn padding_per_stride(m: &Mapping) -> BTreeMap<usize, usize> {
    let mut map = BTreeMap::new();
    let mut stride = 1;
    for axis in axis_leaves(m) {
        map.insert(stride, axis.size());
        stride *= axis.size();
    }
    map
}

/// The axes of a mapping, innermost (right) first, each carrying its padded extent as `size()`.
pub(crate) fn axis_leaves(m: &Mapping) -> Vec<Mapping> {
    let mut axes = Vec::new();
    collect_axis_leaves(m, &mut axes);
    axes
}

/// Padded extent of a division term's axis, or `None` when the term is not an axis boundary.
///
/// Only an axis carries a padded extent. Dividing can split a padded axis into intra-axis sub-terms
/// when an axis between its digits is reduced, and such a sub-term owns no extent to report. A term
/// over adjacent digits starts at its innermost digit's leaf, yet spans every live cell it matched.
pub(crate) fn padded_extent_at(padding_per_stride: &BTreeMap<usize, usize>, term: &DivideTerm) -> Option<usize> {
    padding_per_stride
        .get(&term.dividend_stride)
        .map(|&extent| extent.max(term.resize))
}

/// Keep explicit physical boundaries before normalizing each leaf; normalization may combine a
/// padded scalar with its neighbor and erase a boundary retained by the reducer.
fn collect_axis_leaves(m: &Mapping, out: &mut Vec<Mapping>) {
    match m {
        Mapping::Pair { left, right } => {
            collect_axis_leaves(right, out);
            collect_axis_leaves(left, out);
        }
        other => match other.normalize() {
            Mapping::Pair { left, right } => {
                collect_axis_leaves(&right, out);
                collect_axis_leaves(&left, out);
            }
            axis => out.push(axis),
        },
    }
}

/// Retain physical scalar-padding boundaries that exact named-axis division cannot return.
pub(crate) fn retain_padding(
    input: &Mapping,
    output: &Mapping,
    terms: &mut Vec<DivideTerm>,
) -> Result<(), DivideError> {
    let positions = |mapping| {
        axis_leaves(mapping)
            .into_iter()
            .scan(1, |stride, axis| {
                let position = *stride;
                *stride *= axis.size();
                Some((axis, position))
            })
            .collect::<Vec<_>>()
    };
    let input_axes = positions(input);
    for (axis, divisor_stride) in positions(output) {
        if !matches!(&axis, Mapping::Padding { inner, .. } if **inner == Mapping::identity()) {
            continue;
        }
        let (_, dividend_stride) = input_axes
            .iter()
            .find(|(input, stride)| {
                input == &axis
                    && terms
                        .iter()
                        .all(|term| stride.cmp(&term.dividend_stride) == divisor_stride.cmp(&term.divisor_stride))
            })
            .ok_or(DivideError::ExtentsMisaligned)?;
        if !terms.iter().any(|term| term.divisor_stride == divisor_stride) {
            terms.push(DivideTerm {
                dividend_stride: *dividend_stride,
                divisor_stride,
                resize: 1,
                idents: Default::default(),
            });
        }
    }
    terms.sort_by_key(|term| std::cmp::Reverse(term.dividend_stride));
    Ok(())
}

/// Live positions inner to the outermost reduced axis (`InnerTime::SIZE`), the accumulator slots a
/// reducer holds at once, or an error when `post_reduce` does not divide `pre_reduce`.
///
/// The division matches the retained axes; the reduced ones are the gaps between them. Padded
/// extents count, since a padded position still takes its slot.
pub(crate) fn inner_reduce_extent(pre_reduce: &Mapping, post_reduce: &Mapping) -> Result<usize, DivideError> {
    if pre_reduce.normalize() == post_reduce.normalize() {
        return Ok(1);
    }
    let mut division_terms = crate::config_divide_exact(DivideInput {
        dividend: pre_reduce.clone(),
        divisor: post_reduce.clone(),
    })?
    .into_iter()
    .collect();
    retain_padding(pre_reduce, post_reduce, &mut division_terms)?;
    let padded_extents = padding_per_stride(pre_reduce);
    // A sub-term is dropped before the walk, not skipped inside it: the walk compares adjacent
    // boundaries, and skipping in place would still compare across the sub-term.
    let boundaries: Vec<(&DivideTerm, usize)> = division_terms
        .iter()
        .filter_map(|term| padded_extent_at(&padded_extents, term).map(|extent| (term, extent)))
        .collect();
    let dividend_end = |&(term, extent): &(&DivideTerm, usize)| term.dividend_stride * extent;
    Ok(if boundaries.is_empty() {
        // All axes reduced.
        1
    } else if dividend_end(&boundaries[0]) < pre_reduce.size() {
        // The outermost axis was reduced, so everything below the top is inner to the reduce.
        post_reduce.size()
    } else {
        // The outermost retained factor reaches the top; walk outer-to-inner to the first gap between
        // adjacent retained terms (the reduce boundary), else nothing is inner to the reduce.
        boundaries
            .windows(2)
            .find(|w| dividend_end(&w[1]) != w[0].0.dividend_stride)
            .map_or(1, |w| w[0].0.divisor_stride)
    })
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::{Ident, PaddingKind};

    use super::*;

    #[test]
    fn retains_outer_padding() {
        let row = Mapping::Symbol {
            symbol: Ident::R,
            size: 1,
        }
        .padding(16, PaddingKind::Top);
        let reduced = Mapping::Symbol {
            symbol: Ident::K,
            size: 4,
        };
        let kept = Mapping::Symbol {
            symbol: Ident::N,
            size: 12,
        };
        for (input, output, expected) in [
            (
                row.clone().pair(reduced.clone()).pair(kept.clone()),
                row.clone().pair(kept.clone()),
                12,
            ),
            (
                reduced.clone().pair(row.clone()).pair(kept.clone()),
                row.clone().pair(kept.clone()),
                192,
            ),
            (kept.clone().pair(reduced).pair(row.clone()), kept.pair(row), 16),
        ] {
            assert_eq!(inner_reduce_extent(&input, &output), Ok(expected));
        }
    }

    /// A kept axis written as two adjacent digits outside the reduced axis holds no slot itself:
    /// the slots are the rows inner to the reduction, as with the axis written whole.
    #[test]
    fn split_outer_digits_hold_no_slots() {
        let output = Mapping::Symbol {
            symbol: Ident::N,
            size: 64,
        };
        let reduced = Mapping::Symbol {
            symbol: Ident::K,
            size: 21,
        };
        let row = Mapping::Symbol {
            symbol: Ident::R,
            size: 16,
        };
        let split = output.clone().stride(8).pair(output.clone().modulo(8));
        for kept in [output, split] {
            let input = kept.clone().pair(reduced.clone()).pair(row.clone());
            assert_eq!(inner_reduce_extent(&input, &kept.pair(row.clone())), Ok(16));
        }
    }

    #[test]
    fn unchanged_mapping_is_unit_extent() {
        let row = Mapping::Symbol {
            symbol: Ident::R,
            size: 1,
        }
        .padding(16, PaddingKind::Top);
        let kept = Mapping::Symbol {
            symbol: Ident::N,
            size: 12,
        };
        let mapping = kept.pair(row);
        assert_eq!(inner_reduce_extent(&mapping, &mapping), Ok(1));
    }
}
