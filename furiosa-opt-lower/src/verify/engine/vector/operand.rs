//! VRF-operand verification: what the VRF indexer can address for one VE access.
//!
//! One access is `Packet` elements per way (8 in `Way8`, 4 in `Way4`) and must come from ONE indexer
//! step, which reads either a **broadcast** (one address feeds every lane) or a **contiguous** run
//! (`Packet` consecutive VRF elements). Anything else has no encoding (the ISA's
//! `ReadIndexerOpMode::{BroadcastRead, ContiguousRead}`). This is the frontend twin of
//! `make_vrf_sequencer`'s tail check in the backend, which otherwise reports the same shape as an
//! internal error long after the DSL has accepted it.
//!
//! The second rule here, that every named axis of the operand be one the stream matches, is the
//! DSL's policy rather than the hardware's; `IntoBranchedOperand` in `furiosa-opt-std` states both
//! alongside the partition rule its own signature carries.

use furiosa_mapping::{Mapping, MappingExt, PaddingKind, SequencerError, SequencerMode, sequence};

use crate::verify::{HALF_FLIT_ELEMENTS, ONE_FLIT_ELEMENTS};

/// Why the stream cannot read the operand at all, ahead of any indexer question.
///
/// A shape disagreement lands on [`Unread`](Self::Unread) rather than on a "missing axis" cause: the
/// read carves the operand against the stream, so a stream segment the operand lacks broadcasts, and
/// what fails is the operand's own axes going unmatched.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnreadableCause {
    /// Named axes of the operand the stream never matched, so the cells they index go unread. A
    /// broadcast or a pad left over is not one of these: a read may leave those alone.
    #[error(
        "the stream leaves operand axes unmatched: {}. A VE op matches every named axis of its \
         operand against `Time` x `Packet`, so either drop the axis from the operand, which reads it \
         as a broadcast, or widen the stream to cover it. This is the DSL's rule and stricter than \
         the compiler's, so that a half-read operand cannot pass for a working kernel.",
        .0.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
    )]
    Unread(Vec<Mapping>),
    /// A stream segment matched no operand axis, so the read cannot carve the operand at all.
    #[error(
        "part of the stream matches no operand axis: an operand's axes must be the stream's, \
         regrouped, or absent from it entirely (which reads as a broadcast)"
    )]
    Unmatched,
    /// The operand carries a `Bottom` pad, the marker a write hole leaves, so it is not a fully
    /// written register and no read of it is defined.
    #[error("the operand carries a write hole (`Bottom` padding), which a VE op may not read")]
    WriteHole,
}

/// Why a VRF tensor cannot be read as a VE operand.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VrfOperandError {
    /// The stream does not read the operand.
    #[error("VRF operand {element} is not readable by the stream {stream}: {cause}")]
    Unreadable {
        /// The operand's `Element` mapping.
        element: Mapping,
        /// The stream's `[Time, Packet]`.
        stream: Mapping,
        /// Which way the read failed.
        cause: UnreadableCause,
    },
    /// The packet-sized access is neither one broadcast address nor one contiguous run.
    #[error(
        "VRF operand {element} cannot feed the {}-element packet {packet}: the indexer reads one \
         access as a single broadcast address or as {} consecutive elements, and this operand gives \
         it neither. Write the operand with the packet axes innermost and contiguous (change the \
         `collect` / `to_vrf` that produced it), or drop them from it entirely (broadcast). \
         `reshape` cannot fix this: it may not permute axes.",
        packet.size(),
        packet.size()
    )]
    PacketNotIndexable {
        /// The operand's `Element` mapping.
        element: Mapping,
        /// The stream's `Packet`, whose size is the access window.
        packet: Mapping,
    },
}

/// Judges an operand against both rules the [module docs](self) state.
///
/// Split first, then sequence, which is the order the backend's `make_vrf_sequencer` takes: peel one
/// access off the operand's innermost end and judge it against the packet, then let the remainder
/// answer to `Time`. `packet` is one of the two ALU access widths, which is what the assert pins:
/// the DSL's way already tracks that width, and no other has an indexer.
pub fn config_vrf_operand(element: &Mapping, time: &Mapping, packet: &Mapping) -> Result<(), VrfOperandError> {
    let access = packet.size();
    assert!(
        matches!(access, HALF_FLIT_ELEMENTS | ONE_FLIT_ELEMENTS),
        "the access window is the ALU width ({HALF_FLIT_ELEMENTS} or {ONE_FLIT_ELEMENTS}), \
         got a {access}-element packet {packet}"
    );
    let unreadable = |cause| VrfOperandError::Unreadable {
        element: element.clone(),
        stream: time.clone().pair(packet.clone()),
        cause,
    };
    let not_indexable = || VrfOperandError::PacketNotIndexable {
        element: element.clone(),
        packet: packet.clone(),
    };

    // Judged on the whole operand, since only the outer half reaches `sequence` below and a hole in
    // the access itself would go unseen there.
    if has_write_hole(element) {
        return Err(unreadable(UnreadableCause::WriteHole));
    }

    // Cut one access off the operand's innermost end: `elem_packet` is what one access sees and
    // `elem_time` is what the indexer steps through between accesses, so `m![D, H]` under an
    // 8-element access cuts into `m![D, H / 8]` and `m![H % 8]`. An operand that does not hold a
    // whole number of accesses has no such cut, and no read either.
    if !element.size().is_multiple_of(access) {
        return Err(not_indexable());
    }
    let (elem_time, elem_packet) = element.split_at(access);

    // The indexer's two reads: the access either IS the packet (contiguous) or shares no axis with it
    // (broadcast, one address for every lane). A cut that half-matches has no encoding.
    let packet_axes = packet.idents();
    let contiguous = elem_packet.normalize() == packet.normalize();
    let broadcast = elem_packet.idents().iter().all(|axis| !packet_axes.contains(axis));
    if !contiguous && !broadcast {
        return Err(not_indexable());
    }

    sequence(&[&elem_time], &[time], SequencerMode::Read).map_err(|error| {
        unreadable(match error {
            SequencerError::Unconsumed(memories) => UnreadableCause::Unread(memories.into_iter().collect()),
            SequencerError::StreamUnmatchedSegment => UnreadableCause::Unmatched,
            SequencerError::InputBottomPadding => UnreadableCause::WriteHole,
        })
    })?;
    Ok(())
}

/// Whether the mapping carries a `Bottom` pad anywhere, which is the marker a `view_mut` write hole
/// leaves behind.
fn has_write_hole(mapping: &Mapping) -> bool {
    match mapping {
        Mapping::Padding {
            kind: PaddingKind::Bottom,
            ..
        } => true,
        Mapping::Stride { inner, .. }
        | Mapping::Modulo { inner, .. }
        | Mapping::Resize { inner, .. }
        | Mapping::Padding { inner, .. } => has_write_hole(inner),
        Mapping::Pair { left, right } => has_write_hole(left) || has_write_hole(right),
        Mapping::Symbol { .. } | Mapping::Broadcast { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    // Glob import: the `m!` macro expands to DSL type-level structs that must all be in scope.
    use furiosa_mapping::*;

    use super::*;

    axes![H = 32, G = 4, D = 8, P = 4];

    /// The logical coordinate a live cell reads, per axis. A term `symbol // stride % modulo` at
    /// digit `d` contributes `d * stride` to that symbol, so `m![W / 2, W % 2]` and `m![W]` decode
    /// the same physical position to the same `W`. `None` is a cell that is padding or out of bounds.
    fn coords<Map: M>(offset: usize) -> Option<BTreeMap<Ident, usize>> {
        let Cell::Index(index) = Map::map(offset) else {
            return None;
        };
        let mut axes = BTreeMap::new();
        for (term, digit) in index.0.iter() {
            let Atom::Symbol { symbol, .. } = &term.inner else {
                panic!("these mappings name symbols, not composites")
            };
            *axes.entry(*symbol).or_insert(0) += digit * term.stride;
        }
        Some(axes)
    }

    /// Where every stream position reads the operand, walked from the MAPPINGS alone: decode the
    /// stream cell's coordinates, then take the operand offset whose own coordinates that cell
    /// supplies. `None` is a position the operand cannot answer, which is the stream reading an axis
    /// the operand does not carry.
    ///
    /// This is what makes the assertion below an oracle rather than a restatement: it shares nothing
    /// with the rule, which walks no position and reads no cell. It works in logical coordinates
    /// where the rule works in `sequence`'s strides, so a rule that folded the wrong axes would still
    /// return `Ok` and this would not.
    fn simulate<Element: M, Time: M, Packet: M>() -> Vec<Option<usize>> {
        let live: Vec<(BTreeMap<Ident, usize>, usize)> = (0..Element::SIZE)
            .filter_map(|offset| coords::<Element>(offset).map(|axes| (axes, offset)))
            .collect();
        (0..Time::SIZE * Packet::SIZE)
            .map(|position| {
                // The packet is the stream's innermost run, which is also the access window below.
                let mut want = coords::<Time>(position / Packet::SIZE)?;
                for (axis, value) in coords::<Packet>(position % Packet::SIZE)? {
                    // Summed, not overwritten: one axis may be split across `Time` and `Packet`.
                    *want.entry(axis).or_insert(0) += value;
                }
                live.iter()
                    .find(|(axes, _)| axes.iter().all(|(axis, value)| want.get(axis) == Some(value)))
                    .map(|(_, offset)| *offset)
            })
            .collect()
    }

    /// Checks the rule's verdict against [`simulate`]: chunk the operand offsets into accesses and
    /// judge each the way the hardware does (one address for every lane, or consecutive lanes).
    ///
    /// Returns the rule's verdict, so a rejection test pins the error without spelling its mappings a
    /// second time.
    #[track_caller]
    fn assert_verdict<Element: M, Time: M, Packet: M>(legal: bool) -> Result<(), VrfOperandError> {
        let verdict = config_vrf_operand(&Element::to_value(), &Time::to_value(), &Packet::to_value());
        assert_eq!(verdict.is_ok(), legal, "the rule says {verdict:?}");

        let offsets = simulate::<Element, Time, Packet>();
        let simulated = offsets.iter().all(Option::is_some)
            && offsets.chunks(Packet::SIZE).all(|access| {
                let access: Vec<usize> = access.iter().map(|offset| offset.expect("live")).collect();
                access.iter().all(|&offset| offset == access[0])
                    || access
                        .iter()
                        .enumerate()
                        .all(|(lane, &offset)| offset == access[0] + lane)
            });
        assert_eq!(simulated, legal, "the simulation disagrees with the rule: {offsets:?}");
        verdict
    }

    /// A shape the rule accepts, agreed on by the simulation.
    #[track_caller]
    fn assert_legal<Element: M, Time: M, Packet: M>() {
        assert_verdict::<Element, Time, Packet>(true).expect("asserted above");
    }

    /// A shape the rule refuses, agreed on by the simulation, and the error it refuses with.
    #[track_caller]
    fn assert_illegal<Element: M, Time: M, Packet: M>() -> VrfOperandError {
        assert_verdict::<Element, Time, Packet>(false).expect_err("asserted above")
    }

    /// The common operand: the packet reads the operand's innermost axis, one access = 8 consecutive
    /// elements.
    #[test]
    fn contiguous_packet_reads() {
        assert_legal::<m![H], m![H / 8], m![H % 8]>();
    }

    /// A per-time scalar: the operand has no packet axis at all, so one address feeds every lane.
    #[test]
    fn broadcast_packet_reads() {
        assert_legal::<m![D], m![D, H / 8], m![H % 8]>();
    }

    /// A `to_vrf` stores whole flits, so an operand is a whole number of accesses; one that is not
    /// has no access to cut and is refused ahead of either read. The simulation calls this legal,
    /// since four cells do broadcast, so this is the rule's own check like [`UnreadableCause::Unread`]
    /// and it is asserted directly rather than through [`assert_illegal`].
    #[test]
    fn operand_shorter_than_one_access_rejects() {
        let error = config_vrf_operand(
            &<m![G]>::to_value(),
            &<m![G, D / 8]>::to_value(),
            &<m![D % 8]>::to_value(),
        )
        .unwrap_err();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
    }

    /// An operand with no axis the packet names, padding included, is that same broadcast: what makes
    /// it one is the absent axis, not the padding.
    #[test]
    fn operand_without_packet_axis_broadcasts() {
        assert_legal::<m![1 # 8], m![H / 8], m![H % 8]>();
    }

    /// Two `P` rows in one access, packed: the second row starts where the first ends, so the access
    /// is one contiguous run.
    #[test]
    fn packet_spanning_packed_rows_reads() {
        assert_legal::<m![G, P], m![G / 2], m![G % 2, P]>();
    }

    /// A 4-way op reads 4 lanes, so the same rows in a `P`-padded operand stay legal: the pad sits
    /// outside the access. The window is the packet, not a fixed 8.
    #[test]
    fn way4_packet_reads_inside_padded_row() {
        assert_legal::<m![G, P # 8], m![G], m![P]>();
    }

    /// The packet walks the operand's OUTER axis, so consecutive lanes sit `D` elements apart. One
    /// access cannot gather them.
    #[test]
    fn strided_packet_rejects() {
        let error = assert_illegal::<m![G, D], m![D], m![G]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
    }

    /// Half the access broadcasts and half reads live elements. Each half alone is indexable; the mix
    /// is not, since the indexer picks one mode per access.
    #[test]
    fn mixed_broadcast_and_live_lanes_reject() {
        let error = assert_illegal::<m![D], m![D / 4], m![D % 4, 2]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
    }

    /// The same two rows in a `P`-padded operand read by an 8-wide packet: each row occupies 8
    /// elements, so the second row's lanes sit 8 apart, not 4. The rule reads the PADDED stride,
    /// which is the one the indexer walks.
    #[test]
    fn padding_breaks_contiguity() {
        let error = assert_illegal::<m![G, P # 8], m![G / 2], m![G % 2, P]>();
        assert!(matches!(&error, VrfOperandError::PacketNotIndexable { .. }), "{error}");
        // This wording is pinned in three places: here, the answer key that catches the emulator's
        // panic for `negative::vector_engine::ve_vrf_strided_packet`, and that fixture's `reason` in
        // `snapshot.toml`. Keep the three in step.
        assert!(
            error.to_string().contains("cannot feed the 8-element packet"),
            "{error}"
        );
    }

    /// A packet built from an axis the operand does not have is a shape error, not an indexer limit.
    /// It surfaces as the operand's OWN axis going unread, because the packet axis simply broadcasts.
    #[test]
    fn packet_axis_the_operand_lacks_rejects() {
        let error = assert_illegal::<m![H], m![1], m![D]>();
        assert!(
            matches!(
                &error,
                VrfOperandError::Unreadable {
                    cause: UnreadableCause::Unread(_),
                    ..
                }
            ),
            "{error}"
        );
    }

    /// A `vector_narrow_split` puts the two half-flits on an axis just OUTSIDE the packet, so an
    /// operand written contiguous across all 8 stays legal under a 4-wide packet: the back half is
    /// the indexer's next step, not part of this access, and the half-flit axis is just another
    /// indexer axis.
    #[test]
    fn narrow_split_half_flit_reads() {
        assert_legal::<m![D], m![D / 4], m![D % 4]>();
    }

    /// `Time` carries no indexability rule, so it may walk the operand in any order: here it steps
    /// the operand's INNER axis first and its outer one second, while the packet still reads eight
    /// consecutive cells.
    #[test]
    fn time_walking_the_operand_out_of_order_reads() {
        assert_legal::<m![G, H], m![H / 8, G], m![H % 8]>();
    }

    /// A `Bottom` pad is the marker a write hole leaves, so an operand carrying one is not a register
    /// a read is defined on. The other `Unreadable` cause, and the one that fails before any indexer
    /// question.
    #[test]
    fn write_hole_operand_rejects() {
        let error = config_vrf_operand(
            &<m![G #{!} 8]>::to_value(),
            &<m![1]>::to_value(),
            &<m![G #{!} 8]>::to_value(),
        )
        .unwrap_err();
        assert!(
            matches!(
                &error,
                VrfOperandError::Unreadable {
                    cause: UnreadableCause::WriteHole,
                    ..
                }
            ),
            "{error}"
        );
    }

    /// An operand the stream reads only half of is refused rather than half-read, which is the
    /// policy [`UnreadableCause::Unread`] spells out.
    #[test]
    fn partially_read_operand_rejects() {
        let error = assert_illegal::<m![G, D], m![D / 8], m![D % 8]>();
        assert!(
            matches!(
                &error,
                VrfOperandError::Unreadable {
                    cause: UnreadableCause::Unread(_),
                    ..
                }
            ),
            "{error}"
        );
    }
}
