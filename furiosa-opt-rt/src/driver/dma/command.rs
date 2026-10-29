//! One PDMA channel command: a scatter list the driver moves in one direction, from page-locked
//! memory it already holds by its dma-buf, or from pageable memory it pins for the command.

use std::fs::File;
use std::mem::transmute;
use std::ops::Range;
use std::os::fd::AsRawFd;

use io_uring::{opcode::UringCmd16, squeue::Entry, types::Fd};
use smallvec::{SmallVec, smallvec};

use crate::driver::kernel::{
    DmabufCommand, ScatterCommand, Segment, URING_CMD_DMABUF_READ, URING_CMD_DMABUF_WRITE, URING_CMD_SCATTER_READ,
    URING_CMD_SCATTER_WRITE, node,
};
use crate::pinned::DmaBuf;
use crate::{Error, Result};
use std::sync::Arc;

/// The bytes one driver request moves at most: the driver rejects requests totaling 4 GiB, and
/// aligned offsets preserve host page alignment.
const LIMIT: usize = (1_usize << 32) - 4096;

/// Segments between `host` memory and `chip`; the command stays at a fixed address until
/// completion, since the driver reads its scatter list in place.
pub(crate) struct Command {
    chip: u8,
    direction: Direction,
    host: Memory,
    segments: SmallVec<[Segment; 1]>,
    /// The bytes of every segment, at most [`LIMIT`].
    len: usize,
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
            segments: smallvec![direction.segment(at, device, len)],
            len: len as usize,
        }
    }

    /// Whether this request has room for `len` more bytes between `chip` and the memory `dmabuf`
    /// holds, or pageable memory when `None`: it moves at most [`LIMIT`] bytes.
    pub(crate) fn takes(&self, chip: u8, dmabuf: Option<&Arc<DmaBuf>>, len: usize) -> bool {
        let memory = match (&self.host, dmabuf) {
            (Memory::Borrowed, None) => true,
            (Memory::Owned(held), Some(dmabuf)) => Arc::ptr_eq(held, dmabuf),
            (Memory::Borrowed, Some(_)) | (Memory::Owned(_), None) => false,
        };
        memory && self.chip == chip && self.len + len <= LIMIT
    }

    /// Adds `len` bytes at `host` in its host memory and at `device` on its chip, once the request
    /// [`Self::takes`] them.
    pub(crate) fn push(&mut self, host: u64, device: u64, len: u64) {
        self.segments.push(self.direction.segment(host, device, len));
        self.len += len as usize;
    }

    pub(crate) fn encode(&self, channel: &Channel, tag: u64) -> Entry {
        UringCmd16::new(Fd(channel.0.as_raw_fd()), self.opcode())
            .cmd(self.payload())
            .build()
            .user_data(tag)
    }

    pub(crate) fn chunks(len: usize) -> impl ExactSizeIterator<Item = Range<usize>> {
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

    #[cfg(test)]
    pub(crate) fn segments(&self) -> &[Segment] {
        &self.segments
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
        let desc_addr = self.segments.as_ptr().expose_provenance() as u64;
        // Each segment holds a byte at least, and a request at most `LIMIT`, so the count fits.
        let count = u32::try_from(self.segments.len()).expect("fewer segments than bytes");
        // SAFETY: both commands are sixteen bytes of integers with no padding between them.
        unsafe {
            match &self.host {
                Memory::Borrowed => transmute::<ScatterCommand, [u8; 16]>(ScatterCommand {
                    count: u64::from(count),
                    desc_addr,
                }),
                Memory::Owned(lease) => transmute::<DmabufCommand, [u8; 16]>(DmabufCommand {
                    fd: lease.fd(),
                    count,
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

    /// A command takes more segments of its own chip and host memory while its request stays
    /// within the driver's 4 GiB, and its payload counts every segment.
    #[test]
    fn takes_segments_of_its_chip_and_memory_within_the_limit() {
        let (pinned, other) = (crate::test::pinned(), crate::test::pinned());
        let (dmabuf, foreign) = (pinned.dmabuf(), other.dmabuf());
        let mut owned = Command::new(0, Direction::Write, Host::Owned(Arc::clone(dmabuf), 0), 0, 4096);
        assert!(owned.takes(0, Some(dmabuf), LIMIT - 4096));
        assert!(!owned.takes(0, Some(dmabuf), LIMIT - 4095));
        assert!(!owned.takes(1, Some(dmabuf), 1));
        assert!(!owned.takes(0, Some(foreign), 1));
        assert!(!owned.takes(0, None, 1));
        let borrowed = Command::new(0, Direction::Read, Host::Borrowed(16), 0, 4096);
        assert!(borrowed.takes(0, None, 1) && !borrowed.takes(0, Some(dmabuf), 1));
        owned.push(8192, 4096, 4096);
        assert_eq!(u32::from_ne_bytes(owned.payload()[4..8].try_into().unwrap()), 2);
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
            assert_eq!(command.segments[0].src, src);
            assert_eq!(command.opcode(), opcode);
            let payload = command.payload();
            assert_eq!(u64::from_ne_bytes(payload[..8].try_into().unwrap()), header);
        }
    }
}
