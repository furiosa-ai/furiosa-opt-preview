//! Device DMA configuration and host operands, bound before execution.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::DEVICE_MEMORY;

/// The PDMA engine the host lends one cluster's firmware: the runtime builds it per cluster and
/// sends it in a DMA request, and the firmware's PDMA queue checks it before use.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub struct Config {
    /// The PDMA register bank's address in the firmware's address space, 4-byte aligned.
    pub registers: u64,
    /// The device-memory address of the firmware's descriptor chains, one per channel, 64-byte
    /// aligned.
    pub descriptors: u64,
    /// The PDMA channels this cluster owns, bit `n` for channel `n`.
    pub channels: u8,
    /// This chip's device memory as a PDMA address.
    pub local: u64,
    /// Each chip's device memory as a PDMA address, in rank order; zero past `members`.
    pub peers: [u64; super::MAX_CHIPS as usize],
    /// Chips the device spans, 1 to [`MAX_CHIPS`](super::MAX_CHIPS).
    pub members: u8,
}

/// DMA-only host mapping word: bit 63 tags it, bits 32..60 hold a 256-byte header offset,
/// and bits 0..32 hold the byte offset. CPU and tensor-DMA accesses fault on this tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand(u64);

impl Operand {
    pub fn new(address: u64, offset: u32) -> Option<Self> {
        let address = address.checked_sub(DEVICE_MEMORY.start)?;
        (address < DEVICE_MEMORY.end - DEVICE_MEMORY.start && address.is_multiple_of(HEADER_ALIGNMENT)).then_some(Self(
            (1 << 63) | ((address / HEADER_ALIGNMENT) << 32) | u64::from(offset),
        ))
    }

    /// Rejects reserved tag bits before the word can name a host mapping.
    pub fn parse(word: u64) -> Option<Self> {
        (word >> 60 == 8).then_some(Self(word))
    }

    pub fn encode(self) -> u64 {
        self.0
    }

    /// The mapping header's device address and the byte offset within that mapping.
    pub fn decode(self) -> (u64, u32) {
        (
            DEVICE_MEMORY.start + ((self.0 >> 32) & 0x0fff_ffff) * HEADER_ALIGNMENT,
            self.0 as u32,
        )
    }
}

/// Where the bytes of one pinned host allocation lie on the bus: this header, then `entries`
/// [`Segment`]s, in device memory. Pinned memory is contiguous to the host but not on the bus, so
/// device DMA walks this table to turn an offset into the allocation into a bus address.
/// Registration validates nonempty extents whose lengths sum to `len` and retains the table
/// through execution; firmware only checks the requested range.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct SegmentTable {
    pub len: u64,
    pub entries: u64,
}

impl SegmentTable {
    /// The header and segments firmware walks for `len` mapped bytes, the last segment cut at
    /// `len`; `None` unless nonempty, non-wrapping segments cover `len`.
    #[cfg(feature = "alloc")]
    pub fn encode(len: u64, segments: impl IntoIterator<Item = Segment>) -> Option<alloc::vec::Vec<u64>> {
        let mut entries = alloc::vec::Vec::new();
        let mut remaining = len;
        for segment in segments {
            if remaining == 0 {
                break;
            }
            let len = remaining.min(segment.len);
            segment.address.checked_add(len).filter(|_| len != 0)?;
            entries.push(Segment { len, ..segment });
            remaining -= len;
        }
        let header: [u64; 2] = zerocopy::transmute!(Self {
            len,
            entries: entries.len() as u64
        });
        let words = entries
            .into_iter()
            .flat_map(|entry| -> [u64; 2] { zerocopy::transmute!(entry) });
        (remaining == 0).then(|| header.into_iter().chain(words).collect())
    }
}

/// One contiguous device-visible extent of a mapped host allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Segment {
    pub address: u64,
    pub len: u64,
}

/// Alignment of a mapping header in device memory, so its address fits the operand's 28 bits.
const HEADER_ALIGNMENT: u64 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    /// Firmware decodes the host DMA operand the host encodes, and nothing else.
    #[test]
    fn preserves_host_offsets() {
        for (address, word) in [
            (0xc0_0000_0000, 0x8000_0000_0000_0000),
            (0xc7_1234_5600, 0x8712_3456_0000_0000),
            (0xcf_ffff_ff00, 0x8fff_ffff_0000_0000),
        ] {
            for offset in [0, 1, u32::MAX] {
                let operand = Operand::new(address, offset).unwrap().encode();
                assert_eq!(operand, word | u64::from(offset));
                assert_eq!(Operand::parse(operand).map(Operand::decode), Some((address, offset)));
                assert_eq!(Operand::parse(operand | (1 << 60)), None);
            }
        }
        assert_eq!(Operand::new(0xbf_ffff_ff00, 0), None);
        assert_eq!(Operand::new(0xd0_0000_0000, 0), None);
        assert_eq!(Operand::new(0xc0_0000_0001, 0), None);
        assert_eq!(Operand::parse(0xc0_0000_0000), None);
    }

    /// The segment table firmware walks covers exactly the mapped bytes.
    #[cfg(feature = "alloc")]
    #[test]
    fn tables_cover_the_mapped_bytes() {
        let segment = |address, len| Segment { address, len };
        let segments = [segment(0x1000, 8), segment(0x3000, 16), segment(0x5000, 8)];
        assert_eq!(
            SegmentTable::encode(12, segments),
            Some(alloc::vec![12, 2, 0x1000, 8, 0x3000, 4])
        );
        assert_eq!(SegmentTable::encode(40, segments), None);
        assert_eq!(SegmentTable::encode(12, [segment(0x1000, 0)]), None);
        assert_eq!(SegmentTable::encode(12, [segment(u64::MAX, 12)]), None);
    }
}
