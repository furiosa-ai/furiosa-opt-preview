use std::task::{Context, Poll, Waker};

use furiosa_opt_ipc::{Args, MAX_LAUNCH_ARGS};

use super::Operand;
use crate::arch::rngd;
use crate::memory::Kind;
use crate::{Buffer, BufferSlice, Device, Error, HostBufferSlice, Pinned, Result, Topology};

pub(crate) enum Prepared<'a> {
    Inline(Inline<'a>),
    Indirect(Indirect<'a>),
}

impl Prepared<'_> {
    pub(crate) fn belongs_to(&self, device: &Device) -> bool {
        let owner = match self {
            Self::Inline(inline) => inline.device,
            Self::Indirect(indirect) => indirect.device,
        };
        std::ptr::eq(owner, device)
    }

    pub(crate) fn encode<'a>(&self, window: u64, cluster: usize, words: &'a mut [u64]) -> Args<'a> {
        match self {
            Self::Inline(inline) => {
                Operand::encode(inline.operands, window, cluster, words);
                Args::Inline(&words[..inline.operands.len()])
            }
            Self::Indirect(indirect) => Args::Indirect {
                address: rngd::DEVICE_MEMORY.start
                    + window
                    + (indirect.table.addr() + cluster % indirect.clusters * indirect.count as usize * 8) as u64,
                count: indirect.count,
            },
        }
    }
}

pub(crate) struct Inline<'a> {
    device: &'a Device,
    operands: &'a [Operand],
}

impl<'a> Inline<'a> {
    pub(crate) fn new(device: &'a Device, operands: &'a [Operand]) -> Option<Self> {
        (operands.len() <= MAX_LAUNCH_ARGS).then_some(Self { device, operands })
    }
}

pub(crate) struct Indirect<'a> {
    device: &'a Device,
    table: BufferSlice<'a>,
    count: u32,
    clusters: usize,
}

impl<'a> Indirect<'a> {
    pub(crate) fn new(device: &'a Device, table: &'a Buffer, count: usize) -> Result<Self> {
        if !table.belongs_to(&device.allocator) {
            return Err(Error::ForeignBuffer);
        }
        let count = u32::try_from(count).map_err(|_| Error::Invalid("argument count exceeds u32".into()))?;
        let clusters = Topology {
            chips: 1,
            ..device.topology()
        }
        .clusters();
        Ok(Self {
            device,
            table: table.slice(..count as usize * clusters * size_of::<u64>())?,
            count,
            clusters,
        })
    }
}

/// A launch slot's operands for every cluster, uploaded to one DRAM table per chip.
pub(super) struct Table {
    host: Staging,
    table: Buffer,
    written: Vec<u8>,
}

impl Table {
    pub(super) fn new(device: &Device, capacity: usize) -> Result<Option<Self>> {
        if capacity <= MAX_LAUNCH_ARGS {
            return Ok(None);
        }
        let words = capacity
            * Topology {
                chips: 1,
                ..device.topology()
            }
            .clusters();
        Ok(Some(Self {
            host: Staging::zeroed(words * usize::from(device.topology().chips)),
            table: device.alloc(Kind::Dram, words * 8)?,
            written: Vec::with_capacity(words * usize::from(device.topology().chips) * size_of::<u64>()),
        }))
    }

    pub(super) fn encode(&mut self, device: &Device, operands: &[Operand]) -> usize {
        device.driver.encode(operands, self.host.words());
        let words = operands.len()
            * Topology {
                chips: 1,
                ..device.topology()
            }
            .clusters();
        words * usize::from(device.topology().chips) * size_of::<u64>()
    }

    pub(super) fn cached(&self, bytes: usize) -> bool {
        self.written == self.host.bytes(bytes)
    }

    pub(super) fn reuse(&mut self, other: &mut Self, bytes: usize) -> bool {
        if other.written != self.host.bytes(bytes) {
            return false;
        }
        self.swap(other);
        true
    }

    pub(super) fn empty(&self) -> bool {
        self.written.is_empty()
    }

    pub(super) fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.table, &mut other.table);
        std::mem::swap(&mut self.written, &mut other.written);
    }

    pub(super) fn upload<'a>(&'a mut self, device: &'a Device, count: usize) -> Result<Indirect<'a>> {
        let words = count
            * Topology {
                chips: 1,
                ..device.topology()
            }
            .clusters();
        let bytes = words * usize::from(device.topology().chips) * size_of::<u64>();
        if self.written == self.host.bytes(bytes) {
            return Indirect::new(device, &self.table, count);
        }
        let host = self.host.slice(bytes / size_of::<u64>());
        // No launch reads this table now: a slot uploads only while it is idle, and swaps tables
        // only with an idle slot.
        {
            let mut write = std::pin::pin!(device.write([(host, self.table.slice_mut(..words * 8)?)]));
            loop {
                if let Poll::Ready(result) = write.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
                    result?;
                    break;
                }
                // A pinned staging write is the device worker's job; this thread runs it too, so the
                // launch does not depend on another thread.
                device.work().complete(crate::device::Contention::Wait);
                std::thread::yield_now();
            }
        }
        self.written.clear();
        self.written.extend_from_slice(self.host.bytes(bytes));
        Indirect::new(device, &self.table, count)
    }
}

/// Host words of a table: page-locked where the system provides it, else plain memory that the
/// driver transfers synchronously.
enum Staging {
    Pinned(Pinned),
    Pageable(Vec<u64>),
}

impl Staging {
    fn zeroed(words: usize) -> Self {
        Pinned::zeroed(words * size_of::<u64>()).map_or_else(|_| Self::Pageable(vec![0; words]), Self::Pinned)
    }

    fn words(&mut self) -> &mut [u64] {
        match self {
            Self::Pinned(pinned) => bytemuck::cast_slice_mut(pinned),
            Self::Pageable(words) => words,
        }
    }

    fn bytes(&self, len: usize) -> &[u8] {
        match self {
            Self::Pinned(pinned) => &pinned[..len],
            Self::Pageable(memory) => &bytemuck::cast_slice(memory)[..len],
        }
    }

    fn slice(&self, words: usize) -> HostBufferSlice<'_> {
        match self {
            Self::Pinned(pinned) => pinned.slice(..words * size_of::<u64>()),
            Self::Pageable(memory) => bytemuck::cast_slice(&memory[..words]).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An indirect launch cannot name a foreign or undersized device table.
    #[test]
    fn refuses_foreign_or_short_tables() {
        let device = crate::test::device(&[0], 8);
        let foreign = crate::test::device(&[0], 8);
        let small = device.alloc(Kind::Dram, 8).unwrap();
        let other = foreign.alloc(Kind::Dram, 2048).unwrap();
        let refused = [
            matches!(Indirect::new(&device, &small, 128), Err(Error::Invalid(_))),
            matches!(Indirect::new(&device, &other, 128), Err(Error::ForeignBuffer)),
        ];
        assert_eq!(refused, [true, true]);
    }
}
