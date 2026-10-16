use std::task::{Context, Poll, Waker};

use furiosa_opt_ipc::Args;

use super::Operand;
use crate::arch::rngd;
use crate::memory::Kind;
use crate::{Buffer, BufferSlice, Device, Error, HostBufferSlice, Pinned, Result, Topology};

pub(crate) enum Prepared<'a> {
    Inline(Inline<'a>),
    Indirect(Indirect<'a>),
}

pub(crate) struct Inline<'a> {
    device: &'a Device,
    operands: &'a [Operand],
}

pub(crate) struct Indirect<'a> {
    device: &'a Device,
    table: BufferSlice<'a>,
    count: u32,
    clusters: usize,
}

/// A launch slot's operands for every cluster, uploaded to one DRAM table per chip.
pub(super) struct Table {
    host: Staging,
    table: Buffer,
    written: Vec<u8>,
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

impl<'a> Inline<'a> {
    pub(crate) fn new(device: &'a Device, operands: &'a [Operand]) -> Option<Self> {
        Args::fits_inline(operands.len()).then_some(Self { device, operands })
    }
}

impl<'a> Indirect<'a> {
    pub(crate) fn new(device: &'a Device, table: &'a Buffer, count: usize) -> Result<Self> {
        if !table.belongs_to(&device.allocations) {
            return Err(Error::ForeignBuffer);
        }
        let count = u32::try_from(count).map_err(|_| Error::Invalid("argument count exceeds u32".into()))?;
        let clusters = rngd::clusters(Topology {
            chips: 1,
            ..device.topology()
        });
        Ok(Self {
            device,
            table: table.slice(..count as usize * clusters * size_of::<u64>())?,
            count,
            clusters,
        })
    }
}

impl Table {
    pub(super) fn new(device: &Device, capacity: usize) -> Result<Option<Self>> {
        if Args::fits_inline(capacity) {
            return Ok(None);
        }
        let words = capacity
            * rngd::clusters(Topology {
                chips: 1,
                ..device.topology()
            });
        Ok(Some(Self {
            host: Staging::zeroed(words * usize::from(device.topology().chips)),
            table: device.alloc(Kind::Dram, words * 8)?,
            written: Vec::with_capacity(words * usize::from(device.topology().chips) * size_of::<u64>()),
        }))
    }

    pub(super) fn encode(&mut self, device: &Device, operands: &[Operand]) -> usize {
        device.driver.encode(operands, self.host.words());
        let words = operands.len()
            * rngd::clusters(Topology {
                chips: 1,
                ..device.topology()
            });
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
            * rngd::clusters(Topology {
                chips: 1,
                ..device.topology()
            });
        let bytes = words * usize::from(device.topology().chips) * size_of::<u64>();
        if self.written == self.host.bytes(bytes) {
            return Indirect::new(device, &self.table, count);
        }
        let host = self.host.slice(bytes / size_of::<u64>());
        // Admission refuses, not waits: a table a running launch reads is `Busy`.
        device.admit(|| crate::Lease::acquire([(self.table.slice(..words * 8)?, crate::Access::Write)]))?;
        {
            let mut write = std::pin::pin!(device.write([(host, self.table.slice_mut(..words * 8)?)]));
            loop {
                if let Poll::Ready(result) = write.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
                    result?;
                    break;
                }
                std::thread::yield_now();
            }
        }
        self.written.clear();
        self.written.extend_from_slice(self.host.bytes(bytes));
        Indirect::new(device, &self.table, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unchanged operands reuse their device table while a reader holds it.
    #[test]
    fn reuses_table_when_unchanged() {
        // Given: an uploaded argument table still read by a launch.
        let device = crate::test::device(&[0], 8);
        let mut table = Table::new(&device, 128).unwrap().unwrap();
        let operands = (0..128).map(Operand::Value).collect::<Vec<_>>();
        table.encode(&device, &operands);
        table.upload(&device, operands.len()).unwrap();
        let alias = table.table.alias(..).unwrap();
        let mut held = crate::buffer::Holds::new(1);
        held.acquire([(alias.slice(..).unwrap(), crate::Access::Read)]).unwrap();
        // When: another launch binds the identical operands.
        table.encode(&device, &operands);
        let reused = table.upload(&device, operands.len());
        // Then: it requires no conflicting host write.
        assert!(reused.is_ok());
    }

    /// Failed operand changes preserve the last successfully uploaded table.
    #[test]
    fn preserves_table_when_update_refused() {
        // Given: a successfully uploaded table held by a reader.
        let device = crate::test::device(&[0], 8);
        let mut table = Table::new(&device, 128).unwrap().unwrap();
        let operands = (0..128).map(Operand::Value).collect::<Vec<_>>();
        table.encode(&device, &operands);
        table.upload(&device, operands.len()).unwrap();
        let alias = table.table.alias(..).unwrap();
        let mut held = crate::buffer::Holds::new(1);
        held.acquire([(alias.slice(..).unwrap(), crate::Access::Read)]).unwrap();
        let changed = (1..=128).map(Operand::Value).collect::<Vec<_>>();
        // When: changed operands cannot replace the live table.
        table.encode(&device, &changed);
        let refused = matches!(table.upload(&device, changed.len()), Err(Error::Busy));
        // Then: the original operands still need no transfer.
        assert!(refused);
        table.encode(&device, &operands);
        assert!(table.upload(&device, operands.len()).is_ok());
    }

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
