//! Host memory standing in for device memory.

use std::ops::{Range, RangeBounds};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use furiosa_opt_rt::memory::{Kind, Usage};
use furiosa_opt_rt::{self as rt, Access};

use crate::Error;
use crate::backend::Allocation;
use crate::runtime::Region;

mod bytes;
pub use bytes::Bytes;

/// One owner of host memory standing in for a device allocation: `bytes` of each chip's range.
/// Owners split or aliased from one allocation share its memory, so each sees the others' writes.
#[derive(Debug)]
pub struct Buffer {
    memory: Arc<Memory>,
    bytes: Range<usize>,
}

/// One allocation's bytes, every chip's range chip after chip.
#[derive(Debug)]
struct Memory {
    /// The device it was allocated on; `None` for a device function's own tensor.
    device: Option<(Arc<Capacity>, Kind)>,
    chips: usize,
    /// Bytes each chip holds.
    size: usize,
    cells: Mutex<Bytes>,
}

/// A device's memory budget: what its allocations hold of each kind, against what it offers.
#[derive(Debug)]
pub(super) struct Capacity {
    dram: Budget,
    sram: Budget,
}

#[derive(Debug)]
struct Budget {
    limit: usize,
    /// Allocations round up to this many bytes, as the device's allocator places them.
    alignment: usize,
    used: Mutex<usize>,
}

impl Capacity {
    /// `dram` bytes of DRAM and the RNGD's SRAM window per chip.
    pub(super) fn new(dram: usize) -> Self {
        use furiosa_opt_abi::arch::rngd::sram;
        let budget = |limit, alignment| Budget {
            limit,
            alignment,
            used: Mutex::new(0),
        };
        Self {
            dram: budget(dram, 1),
            sram: budget(
                (sram::WINDOW.end - sram::WINDOW.start) as usize,
                sram::ALIGNMENT as usize,
            ),
        }
    }

    fn budget(&self, kind: Kind) -> &Budget {
        match kind {
            Kind::Sram => &self.sram,
            Kind::Dram | Kind::DramShared => &self.dram,
        }
    }

    /// Bytes per chip `kind` allocations can hold in all.
    pub(super) fn capacity(&self, kind: Kind) -> usize {
        self.budget(kind).limit
    }

    /// Free `kind` bytes per chip now; host memory does not fragment, so all of it is one span.
    pub(super) fn usage(&self, kind: Kind) -> Usage {
        let budget = self.budget(kind);
        let available = budget.limit - *budget.lock();
        Usage {
            available,
            largest: available,
        }
    }
}

impl Budget {
    fn lock(&self) -> MutexGuard<'_, usize> {
        self.used.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn rounded(&self, len: usize) -> usize {
        len.next_multiple_of(self.alignment)
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        if let Some((capacity, kind)) = &self.device {
            let budget = capacity.budget(*kind);
            *budget.lock() -= budget.rounded(self.size);
        }
    }
}

impl Buffer {
    /// A fresh zeroed `len`-byte buffer of `kind` on each of `chips` chips of the device whose
    /// budget `capacity` is; [`rt::Error::OutOfMemory`] when the budget cannot hold it.
    pub(super) fn alloc(capacity: &Arc<Capacity>, kind: Kind, chips: usize, len: usize) -> rt::Result<Self> {
        let budget = capacity.budget(kind);
        let mut used = budget.lock();
        if budget.limit - *used < budget.rounded(len) {
            return Err(rt::Error::OutOfMemory { kind, requested: len });
        }
        *used += budget.rounded(len);
        Ok(Self::new(Some((Arc::clone(capacity), kind)), chips, len))
    }

    /// A fresh zeroed `len`-byte buffer on each of `chips` chips that no device owns and no other
    /// buffer aliases, as a compiled program's own memory is.
    pub(super) fn scratch(chips: usize, len: usize) -> Self {
        Self::new(None, chips, len)
    }

    fn new(device: Option<(Arc<Capacity>, Kind)>, chips: usize, size: usize) -> Self {
        let memory = Memory {
            device,
            chips,
            size,
            cells: Mutex::new(vec![0; size * chips].into()),
        };
        Self {
            memory: Arc::new(memory),
            bytes: 0..size,
        }
    }

    /// Copies `host` into `region` for the device whose budget `device` is, chip after chip in
    /// rank order; the caller checked that `host` holds exactly its bytes. Fails as
    /// [`Self::check_device`].
    pub(super) fn write(&mut self, device: &Arc<Capacity>, region: &Region, host: &[u8]) -> Result<(), Error> {
        self.check_device(device)?;
        self.store(region, &host.to_vec().into());
        Ok(())
    }

    /// Copies `region` into `host` for the device whose budget `device` is, chip after chip in
    /// rank order. Fails as [`Self::check_device`].
    pub(super) fn read(&self, device: &Arc<Capacity>, region: &Region, host: &mut [u8]) -> Result<(), Error> {
        self.check_device(device)?;
        host.copy_from_slice(self.load(region).as_ref());
        Ok(())
    }

    /// [`rt::Error::ForeignBuffer`] unless the memory is the device's whose budget `device` is,
    /// or a device function's own.
    pub(super) fn check_device(&self, device: &Arc<Capacity>) -> Result<(), Error> {
        match &self.memory.device {
            Some((owner, _)) if !Arc::ptr_eq(owner, device) => Err(rt::Error::ForeignBuffer.into()),
            _ => Ok(()),
        }
    }

    /// `region`'s bytes, chip after chip in rank order, as a device function reads them. The whole
    /// allocation shares its bytes.
    pub(crate) fn load(&self, region: &Region) -> Bytes {
        let cells = self.lock();
        if self.covers(region) {
            return cells.clone();
        }
        self.ranges(region)
            .flat_map(|range| cells.as_ref()[range].to_vec())
            .collect::<Vec<_>>()
            .into()
    }

    /// Writes `bytes` into `region`, chip after chip in rank order, as a device function writes.
    /// The whole allocation shares `bytes`.
    pub(super) fn store(&self, region: &Region, bytes: &Bytes) {
        let mut cells = self.lock();
        if self.covers(region) {
            *cells = bytes.clone();
            return;
        }
        let ranges = self.ranges(region);
        let cells = cells.as_mut();
        for (range, bytes) in ranges.zip(bytes.as_ref().chunks_exact(region.bytes.len().max(1))) {
            cells[range].copy_from_slice(bytes);
        }
    }

    /// Whether `region` is every byte of every chip of the allocation.
    fn covers(&self, region: &Region) -> bool {
        self.bytes.start + region.bytes.start == 0
            && region.bytes.len() == self.memory.size
            && region.chips.count_ones() as usize == self.memory.chips
    }

    /// Where `region`'s bytes are in the allocation, chip after chip in rank order.
    fn ranges(&self, region: &Region) -> impl Iterator<Item = Range<usize>> + use<> {
        let stride = self.memory.size;
        let start = self.bytes.start + region.bytes.start;
        let (chips, size) = (region.chips, region.bytes.len());
        (0..self.memory.chips)
            .filter(move |chip| chips & 1 << chip != 0)
            .map(move |chip| chip * stride + start..chip * stride + start + size)
    }

    fn lock(&self) -> MutexGuard<'_, Bytes> {
        // Copies are length-checked before they lock, so a poisoned lock still guards whole bytes.
        self.memory.cells.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `bytes` of this owner's range, [`rt::Error::Invalid`] outside it.
    fn within(&self, bytes: impl RangeBounds<usize>) -> rt::Result<Range<usize>> {
        let bytes = std::slice::try_range(bytes, ..self.bytes.len())
            .ok_or_else(|| rt::Error::Invalid(format!("the range is outside the {}-byte buffer", self.bytes.len())))?;
        Ok(self.bytes.start + bytes.start..self.bytes.start + bytes.end)
    }
}

/// [`rt::Error::Busy`] when two of a launch's `uses` overlap and either writes, as a device refuses
/// a launch that would race with itself.
pub(super) fn check_conflicts<'a>(
    uses: impl IntoIterator<Item = (&'a Buffer, &'a Region, Access)>,
) -> Result<(), Error> {
    let uses = uses.into_iter().collect::<Vec<_>>();
    // ponytail: pairwise over a launch's arguments, which are few.
    for (index, &(first, first_region, first_access)) in uses.iter().enumerate() {
        for &(second, second_region, second_access) in &uses[index + 1..] {
            let overlaps = Arc::ptr_eq(&first.memory, &second.memory)
                && first_region.chips & second_region.chips != 0
                && first.bytes.start + first_region.bytes.start < second.bytes.start + second_region.bytes.end
                && second.bytes.start + second_region.bytes.start < first.bytes.start + first_region.bytes.end;
            if overlaps && (first_access == Access::Write || second_access == Access::Write) {
                return Err(rt::Error::Busy.into());
            }
        }
    }
    Ok(())
}

impl Allocation for Buffer {
    fn size(&self) -> usize {
        self.bytes.len()
    }

    fn chips(&self) -> usize {
        self.memory.chips
    }

    /// Where the bytes start in their allocation, which host memory places at zero.
    fn addr(&self) -> usize {
        self.bytes.start
    }

    fn split_at(self, offset: usize) -> Result<(Self, Self), Error> {
        let at = self.within(offset..)?.start;
        let head = Self {
            memory: Arc::clone(&self.memory),
            bytes: self.bytes.start..at,
        };
        Ok((
            head,
            Self {
                memory: self.memory,
                bytes: at..self.bytes.end,
            },
        ))
    }

    fn alias(&self, bytes: impl RangeBounds<usize>) -> Result<Self, Error> {
        Ok(Self {
            memory: Arc::clone(&self.memory),
            bytes: self.within(bytes)?,
        })
    }

    /// A host launch writes before it returns, so its bytes are ready at once.
    fn ready(&self) -> impl Future<Output = Result<(), Error>> + Send + use<> {
        std::future::ready(Ok(()))
    }

    fn join(&mut self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.memory, &other.memory) && self.bytes.end == other.bytes.start {
            self.bytes.end = other.bytes.end;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Host;
    use super::*;
    use crate::Topology;
    use crate::backend::{Cpu, Runtime, internal::Backend as _};
    use crate::tensor::memory::DmAllocError;
    use furiosa_mapping::*;

    fn whole(bytes: Range<usize>) -> Region {
        Region { chips: 0b11, bytes }
    }

    fn capacity() -> Arc<Capacity> {
        Arc::new(Capacity::new(4096))
    }

    /// Host HBM exhaustion reports the same shared-memory kind as the NPU allocator.
    #[test]
    fn hbm_exhaustion_names_shared_memory() {
        let device = Host {
            topology: Topology { chips: 2, pes: 8 },
            placement: crate::Placement {
                chips: vec![0, 1],
                pes: 0..8,
            },
            capacity: capacity(),
        };
        let _buffer = Cpu::alloc(&device, 4096).unwrap();
        assert!(matches!(
            Cpu::alloc(&device, 1),
            Err(Error::Device(rt::Error::OutOfMemory {
                kind: Kind::DramShared,
                ..
            }))
        ));
    }

    /// A CPU resident that exceeds the per-slice SRAM limit is refused as on the NPU.
    #[test]
    fn residents_obey_sram_capacity() {
        axes![A = 1048576];
        let device = Host {
            topology: Topology { chips: 2, pes: 8 },
            placement: crate::Placement {
                chips: vec![0, 1],
                pes: 0..8,
            },
            capacity: capacity(),
        };
        assert!(matches!(
            Cpu::alloc_dm::<u8, m![2], m![1], m![1 # 64], m![A]>(&device),
            Err(DmAllocError::Allocation(rt::Error::OutOfMemory {
                kind: Kind::Sram,
                ..
            }))
        ));
    }

    /// Owners split or aliased from one buffer address and see the same host bytes.
    #[test]
    fn aliases_share_bytes() {
        let capacity = capacity();
        let mut buffer = Buffer::alloc(&capacity, Kind::DramShared, 2, 4).unwrap();
        let alias = buffer.alias(1..3).unwrap();
        buffer
            .write(&capacity, &whole(0..4), &[10, 11, 12, 13, 14, 15, 16, 17])
            .unwrap();
        let mut bytes = [0; 4];
        alias.read(&capacity, &whole(0..2), &mut bytes).unwrap();
        assert_eq!(bytes, [11, 12, 15, 16]);
        assert!(matches!(alias.alias(1..3), Err(Error::Device(_))));
        let (head, tail) = buffer.split_at(2).unwrap();
        tail.read(&capacity, &whole(0..2), &mut bytes).unwrap();
        assert_eq!(bytes, [12, 13, 16, 17]);
        assert_eq!([alias.addr(), tail.addr()], [head.addr() + 1, head.addr() + 2]);
    }

    /// Overlapping uses of one launch conflict when either writes; reads, other chips and other
    /// bytes do not.
    #[test]
    fn launch_uses_conflict_only_on_overlapping_writes() {
        let capacity = capacity();
        let buffer = Buffer::alloc(&capacity, Kind::DramShared, 2, 4).unwrap();
        let alias = buffer.alias(2..).unwrap();
        let other = Buffer::alloc(&capacity, Kind::DramShared, 2, 4).unwrap();
        let busy = |uses: &[(&Buffer, Region, Access)]| {
            matches!(
                check_conflicts(uses.iter().map(|(buffer, region, access)| (*buffer, region, *access))),
                Err(Error::Device(rt::Error::Busy))
            )
        };
        let chip = |chips, bytes| Region { chips, bytes };
        assert!(busy(&[
            (&buffer, whole(0..4), Access::Write),
            (&alias, whole(0..1), Access::Read)
        ]));
        assert!(!busy(&[
            (&buffer, whole(0..4), Access::Read),
            (&alias, whole(0..1), Access::Read)
        ]));
        assert!(!busy(&[
            (&buffer, whole(0..2), Access::Write),
            (&alias, whole(0..2), Access::Write)
        ]));
        assert!(!busy(&[
            (&buffer, chip(0b01, 0..4), Access::Write),
            (&alias, chip(0b10, 0..2), Access::Write)
        ]));
        assert!(!busy(&[
            (&buffer, whole(0..4), Access::Write),
            (&other, whole(0..4), Access::Write)
        ]));
    }

    /// Freed memory returns to the device's budget.
    #[test]
    fn freed_memory_returns_to_the_budget() {
        let capacity = capacity();
        let buffer = Buffer::alloc(&capacity, Kind::DramShared, 2, 4096).unwrap();
        let (head, tail) = buffer.split_at(1).unwrap();
        drop(head);
        assert_eq!(capacity.usage(Kind::DramShared).available, 0);
        drop(tail);
        assert_eq!(capacity.usage(Kind::DramShared).available, 4096);
    }
}
