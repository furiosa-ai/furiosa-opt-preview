//! Packets a tag-filtered stream emitted, as every engine after it sees them.

/// A number of stream packets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PacketCount(pub usize);

/// Packets each slice of a tag-filtered stream emitted, from the filter to the commit. A slice may
/// emit more than its `packet_capacity`, which only a slice the commit skips may do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilteredPackets {
    emitted_packets: Vec<PacketCount>,
    packet_capacity: PacketCount,
}

/// A slice the commit writes that emits more packets than its filter's output holds. The device
/// hangs once such a slice overflows by four packets (Arena 329481), so every model rejects any
/// overflow. A slice the commit skips may overflow (Arena 329504).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "slice {slice} emits {} packets past its filter's output of {}; the kernel must keep at most that many \
     survivors per slice",
    emitted_packets.0 - packet_capacity.0,
    packet_capacity.0
)]
pub struct PacketOverflow {
    /// The first written slice that overflows.
    pub slice: usize,
    /// Packets that slice emits.
    pub emitted_packets: PacketCount,
    /// Packets the filter's output holds per slice.
    pub packet_capacity: PacketCount,
}

impl FilteredPackets {
    /// What a filter emits into `packet_capacity` packets per slice, when each slice emits
    /// `emitted_packets`.
    pub fn new(emitted_packets: impl IntoIterator<Item = PacketCount>, packet_capacity: PacketCount) -> Self {
        assert!(
            packet_capacity.0 > 0,
            "a filter emits into at least one packet per slice"
        );
        Self {
            emitted_packets: emitted_packets.into_iter().collect(),
            packet_capacity,
        }
    }

    /// Whether every slice emits exactly its `packet_capacity`, like a `FixedPackets` stream.
    pub fn is_full(&self) -> bool {
        self.emitted_packets
            .iter()
            .all(|&emitted| emitted == self.packet_capacity)
    }

    /// Packets each slice holds when full.
    pub fn packet_capacity(&self) -> PacketCount {
        self.packet_capacity
    }

    /// Packets a commit writes to each slice, zero where `is_written` says it skips the slice.
    /// Rejects a written slice that emits past `packet_capacity`.
    pub fn written_packets(&self, is_written: impl Fn(usize) -> bool) -> Result<Vec<PacketCount>, PacketOverflow> {
        (self.emitted_packets.iter().enumerate())
            .map(
                |(slice, &emitted)| match (is_written(slice), emitted > self.packet_capacity) {
                    (false, _) => Ok(PacketCount(0)),
                    (true, false) => Ok(emitted),
                    (true, true) => Err(PacketOverflow {
                        slice,
                        emitted_packets: emitted,
                        packet_capacity: self.packet_capacity,
                    }),
                },
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_skipped_slice_may_emit_past_its_capacity() {
        let filtered = FilteredPackets::new([PacketCount(1), PacketCount(3)], PacketCount(2));
        assert_eq!(
            filtered.written_packets(|slice| slice == 0),
            Ok(vec![PacketCount(1), PacketCount(0)])
        );
        assert_eq!(
            filtered.written_packets(|_| true),
            Err(PacketOverflow {
                slice: 1,
                emitted_packets: PacketCount(3),
                packet_capacity: PacketCount(2),
            })
        );
    }
}
