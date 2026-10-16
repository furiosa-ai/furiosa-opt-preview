//! One PDMA channel command: a segment the driver moves in one direction, from page-locked
//! memory it already holds by its dma-buf, or from pageable memory it pins for the command.

use std::fs::File;
use std::ops::Range;
use std::os::fd::AsRawFd;

use io_uring::{opcode::UringCmd16, squeue::Entry, types::Fd};

use crate::Result;
use crate::arch::rngd::{
    Segment, URING_CMD_DMABUF_READ, URING_CMD_DMABUF_WRITE, URING_CMD_SCATTER_READ, URING_CMD_SCATTER_WRITE, node,
};
use crate::driver::map::open_rw;
use crate::pinned::HostLease;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Write,
    Read,
}

impl Direction {
    /// `len` bytes between host address `host` and device address `device`, in this direction.
    pub(crate) fn segment(self, host: u64, device: u64, len: u64) -> Segment {
        let (src, dst) = match self {
            Self::Write => (host, device),
            Self::Read => (device, host),
        };
        Segment { src, dst, len }
    }
}

/// Where a command's host bytes are.
#[derive(Debug)]
pub(crate) enum Host {
    /// Any process memory at this address; the driver pins its pages for the command.
    Borrowed(u64),
    /// Page-locked memory at this offset of the dma-buf the driver takes by its descriptor.
    Owned(HostLease, u64),
}

enum Memory {
    Borrowed,
    Owned(HostLease),
}

/// One segment between `host` memory and `chip`; its command stays at a fixed address until completion.
pub(crate) struct Command {
    pub(crate) chip: u8,
    pub(crate) direction: Direction,
    host: Memory,
    segment: Segment,
}

impl Command {
    pub(crate) fn chunks(len: usize) -> impl ExactSizeIterator<Item = Range<usize>> {
        // The driver rejects requests totaling 4 GiB; aligned offsets preserve host page alignment.
        const LIMIT: usize = (1_usize << 32) - 4096;
        (0..len)
            .step_by(LIMIT)
            .map(move |start| start..start + LIMIT.min(len - start))
    }

    /// Moves `len` bytes between `host` and the chip's PDMA address `device`, in `direction`.
    pub(crate) fn new(chip: u8, direction: Direction, host: Host, device: u64, len: u64) -> Self {
        let (host, at) = match host {
            Host::Borrowed(at) => (Memory::Borrowed, at),
            Host::Owned(lease, at) => (Memory::Owned(lease), at),
        };
        Self {
            chip,
            direction,
            host,
            segment: direction.segment(at, device, len),
        }
    }

    pub(crate) fn borrowed(&self) -> bool {
        matches!(self.host, Memory::Borrowed)
    }

    pub(crate) fn encode(&self, channel: &Channel, tag: u64) -> Entry {
        UringCmd16::new(Fd(channel.0.as_raw_fd()), self.opcode())
            .cmd(self.payload())
            .build()
            .user_data(tag)
    }

    fn opcode(&self) -> u32 {
        match (&self.host, self.direction) {
            (Memory::Borrowed, Direction::Write) => URING_CMD_SCATTER_WRITE,
            (Memory::Borrowed, Direction::Read) => URING_CMD_SCATTER_READ,
            (Memory::Owned(_), Direction::Write) => URING_CMD_DMABUF_WRITE,
            (Memory::Owned(_), Direction::Read) => URING_CMD_DMABUF_READ,
        }
    }

    /// The sixteen command bytes: `{count: u64, desc_addr: u64}` for pageable memory, and
    /// `{fd: i32, count: u32, desc_addr: u64}` for a dma-buf.
    fn payload(&self) -> [u8; 16] {
        let mut payload = [0; 16];
        let descriptor = (std::ptr::from_ref(&self.segment).expose_provenance() as u64).to_ne_bytes();
        match &self.host {
            Memory::Borrowed => {
                payload[..8].copy_from_slice(&1_u64.to_ne_bytes());
            }
            Memory::Owned(lease) => {
                payload[..4].copy_from_slice(&lease.fd().to_ne_bytes());
                payload[4..8].copy_from_slice(&1_u32.to_ne_bytes());
            }
        }
        payload[8..].copy_from_slice(&descriptor);
        payload
    }
}

pub(crate) struct Channel(File);

impl Channel {
    pub(crate) fn open(chip: u8, channel: u8) -> Result<Self> {
        open_rw(&node(chip, &format!("ch{channel}"))).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Driver requests stay below 4 GiB while covering every byte, including an unaligned tail.
    #[test]
    fn covers_large_transfers_with_bounded_requests() {
        for len in [0, 4096, (1_usize << 32) - 1, 1_usize << 32, 7247757312, 48_usize << 30] {
            let mut end = 0;
            for bytes in Command::chunks(len) {
                assert_eq!(bytes.start, end);
                assert!(
                    bytes.len() < 1_usize << 32,
                    "driver returns E2BIG for {} bytes",
                    bytes.len()
                );
                end = bytes.end;
            }
            assert_eq!(end, len);
        }
    }

    /// Transfers reach the kernel driver as its io_uring opcodes and command payloads.
    #[test]
    fn encodes_driver_commands() {
        let pinned = crate::test::pinned();
        let backing = pinned.slice(..).backing.unwrap();
        let leases = [backing.retain(), backing.retain()];
        let fd = u64::try_from(leases[0].fd()).unwrap();
        let [writer, reader] = leases;
        for (host, direction, opcode, header, src) in [
            (Host::Borrowed(16), Direction::Write, 1, 1, 16),
            (Host::Borrowed(16), Direction::Read, 0, 1, 32),
            (Host::Owned(writer, 16), Direction::Write, 4, fd | 1 << 32, 16),
            (Host::Owned(reader, 16), Direction::Read, 3, fd | 1 << 32, 32),
        ] {
            let command = Command::new(0, direction, host, 32, 48);
            assert_eq!(command.segment.src, src);
            assert_eq!(command.opcode(), opcode);
            let payload = command.payload();
            assert_eq!(u64::from_ne_bytes(payload[..8].try_into().unwrap()), header);
        }
    }
}
