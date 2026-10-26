//! One PDMA channel command: a segment the driver moves in one direction, from page-locked
//! memory it already holds by its dma-buf, or from pageable memory it pins for the command.

use std::fs::File;
use std::mem::transmute;
use std::ops::Range;
use std::os::fd::AsRawFd;

use io_uring::{opcode::UringCmd16, squeue::Entry, types::Fd};

use crate::driver::kernel::{
    DmabufCommand, ScatterCommand, Segment, URING_CMD_DMABUF_READ, URING_CMD_DMABUF_WRITE, URING_CMD_SCATTER_READ,
    URING_CMD_SCATTER_WRITE, node,
};
use crate::pinned::DmaBuf;
use crate::{Error, Result};
use std::sync::Arc;

/// One segment between `host` memory and `chip`; its command stays at a fixed address until completion.
pub(crate) struct Command {
    chip: u8,
    direction: Direction,
    host: Memory,
    segment: Segment,
}

impl Command {
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

    pub(crate) fn encode(&self, channel: &Channel, tag: u64) -> Entry {
        UringCmd16::new(Fd(channel.0.as_raw_fd()), self.opcode())
            .cmd(self.payload())
            .build()
            .user_data(tag)
    }

    pub(crate) fn chunks(len: usize) -> impl ExactSizeIterator<Item = Range<usize>> {
        // The driver rejects requests totaling 4 GiB; aligned offsets preserve host page alignment.
        const LIMIT: usize = (1_usize << 32) - 4096;
        (0..len)
            .step_by(LIMIT)
            .map(move |start| start..start + LIMIT.min(len - start))
    }

    /// The kernel driver index of the chip the command moves bytes on.
    pub(crate) fn chip(&self) -> u8 {
        self.chip
    }

    pub(crate) fn direction(&self) -> Direction {
        self.direction
    }

    pub(crate) fn borrowed(&self) -> bool {
        matches!(self.host, Memory::Borrowed)
    }

    fn opcode(&self) -> u32 {
        match (&self.host, self.direction) {
            (Memory::Borrowed, Direction::Write) => URING_CMD_SCATTER_WRITE,
            (Memory::Borrowed, Direction::Read) => URING_CMD_SCATTER_READ,
            (Memory::Owned(_), Direction::Write) => URING_CMD_DMABUF_WRITE,
            (Memory::Owned(_), Direction::Read) => URING_CMD_DMABUF_READ,
        }
    }

    /// The sixteen command bytes: a [`ScatterCommand`] for pageable memory, a [`DmabufCommand`]
    /// for a dma-buf.
    fn payload(&self) -> [u8; 16] {
        let desc_addr = std::ptr::from_ref(&self.segment).expose_provenance() as u64;
        // SAFETY: both commands are sixteen bytes of integers with no padding between them.
        unsafe {
            match &self.host {
                Memory::Borrowed => transmute::<ScatterCommand, [u8; 16]>(ScatterCommand { count: 1, desc_addr }),
                Memory::Owned(lease) => transmute::<DmabufCommand, [u8; 16]>(DmabufCommand {
                    fd: lease.fd(),
                    count: 1,
                    desc_addr,
                }),
            }
        }
    }
}

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
pub(crate) enum Host {
    /// Any process memory at this address; the driver pins its pages for the command.
    Borrowed(u64),
    /// Page-locked memory at this offset of the dma-buf the driver takes by its descriptor.
    Owned(Arc<DmaBuf>, u64),
}

pub(crate) struct Channel(File);

impl Channel {
    pub(crate) fn open(chip: u8, channel: u8) -> Result<Self> {
        File::options()
            .read(true)
            .write(true)
            .open(node(chip, &format!("ch{channel}")))
            .map_err(|why| Error::Device(format!("{}: {why}", node(chip, &format!("ch{channel}")))))
            .map(Self)
    }
}

enum Memory {
    Borrowed,
    Owned(Arc<DmaBuf>),
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
        let backing = pinned.slice(..).pinned.unwrap();
        let leases = [Arc::clone(backing.dmabuf), Arc::clone(backing.dmabuf)];
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
