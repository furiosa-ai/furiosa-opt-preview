//! Host memory standing in for device memory.

use std::ops::{Range, RangeBounds};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use furiosa_opt_rt as rt;
use furiosa_opt_rt::memory::{Allocations, Kind};

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
    /// Where the device's allocator placed it, released with the last owner; `None` for a device
    /// function's own tensor, which no device places.
    placement: Option<rt::Buffer>,
    chips: usize,
    /// Bytes each chip holds.
    size: usize,
    cells: Mutex<Bytes>,
}

impl Buffer {
    /// A fresh zeroed `len`-byte buffer of `kind` on every chip, placed by `allocations` as on a
    /// device; [`rt::Error::OutOfMemory`] when no free span holds it.
    pub(super) fn alloc(allocations: &Arc<Allocations>, kind: Kind, len: usize) -> rt::Result<Self> {
        let placement = allocations.alloc(kind, len)?;
        let chips = usize::from(placement.topology().chips);
        Ok(Self::new(Some(placement), chips, len))
    }

    /// A fresh zeroed `len`-byte buffer on each of `chips` chips that no device owns and no other
    /// buffer aliases, as a compiled program's own memory is.
    pub(super) fn scratch(chips: usize, len: usize) -> Self {
        Self::new(None, chips, len)
    }

    fn new(placement: Option<rt::Buffer>, chips: usize, size: usize) -> Self {
        let memory = Memory {
            placement,
            chips,
            size,
            cells: Mutex::new(vec![0; size * chips].into()),
        };
        Self {
            memory: Arc::new(memory),
            bytes: 0..size,
        }
    }

    /// [`rt::Error::ForeignBuffer`] unless `allocations` placed the memory, or it is a device
    /// function's own.
    pub(super) fn check_device(&self, allocations: &Arc<Allocations>) -> Result<(), Error> {
        match &self.memory.placement {
            Some(placement) if !placement.belongs_to(allocations) => Err(rt::Error::ForeignBuffer.into()),
            _ => Ok(()),
        }
    }

    /// `region`'s bytes, chip after chip in rank order. A region of the whole allocation shares
    /// its bytes.
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

    /// Writes `bytes`, which the caller checked hold exactly `region`'s, chip after chip in rank
    /// order. A region of the whole allocation shares `bytes`.
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

    /// `region`'s bytes where the device's allocator placed them, on every chip as a launch binds
    /// them; `None` for a device function's own memory, which no device places.
    pub(super) fn placed(&self, region: &Region) -> Option<rt::BufferSlice<'_>> {
        let start = self.bytes.start + region.bytes.start;
        let slice = self.memory.placement.as_ref()?.slice(start..start + region.bytes.len());
        Some(slice.expect("a region lies inside its buffer"))
    }

    /// `bytes` of this owner's range, [`rt::Error::Invalid`] outside it.
    fn within(&self, bytes: impl RangeBounds<usize>) -> rt::Result<Range<usize>> {
        let bytes = std::slice::try_range(bytes, ..self.bytes.len())
            .ok_or_else(|| rt::Error::Invalid(format!("the range is outside the {}-byte buffer", self.bytes.len())))?;
        Ok(self.bytes.start + bytes.start..self.bytes.start + bytes.end)
    }
}

impl Allocation for Buffer {
    fn size(&self) -> usize {
        self.bytes.len()
    }

    fn chips(&self) -> usize {
        self.memory.chips
    }

    /// Where the device placed the allocation, plus this owner's offset in it; a device function's
    /// own memory starts at zero.
    fn addr(&self) -> usize {
        self.memory.placement.as_ref().map_or(0, rt::Buffer::addr) + self.bytes.start
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
    use super::*;
    use crate::backend::Cpu;
    use crate::runtime::{Access, Device, HbmBuffer, Topology};

    fn whole(bytes: Range<usize>) -> Region {
        Region { chips: 0b11, bytes }
    }

    fn allocations() -> Arc<Allocations> {
        Arc::new(Allocations::new(Topology { chips: 2, pes: 8 }))
    }

    /// Host memory is placed as the device's allocator places it: DRAM in whole 512-byte units,
    /// and the span a buffer freed between two live ones leaves is not the largest free span.
    #[test]
    fn places_memory_as_the_device_allocator() {
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let usage = || device.memory().usage();
        let free = usage().available;
        let [_first, second, _third] = [(); 3].map(|()| HbmBuffer::alloc(&device, 1).unwrap());
        assert_eq!(free - usage().available, 3 * 512);
        drop(second);
        assert_eq!(usage().available - usage().largest, 512);
    }

    /// Owners split or aliased from one buffer address and see the same host bytes.
    #[test]
    fn aliases_share_bytes() {
        let buffer = Buffer::alloc(&allocations(), Kind::DramShared, 4).unwrap();
        let alias = buffer.alias(1..3).unwrap();
        buffer.store(&whole(0..4), &vec![10, 11, 12, 13, 14, 15, 16, 17].into());
        assert_eq!(*alias.load(&whole(0..2)), [11, 12, 15, 16]);
        assert!(matches!(alias.alias(1..3), Err(Error::Device(_))));
        let (head, tail) = buffer.split_at(2).unwrap();
        assert_eq!(*tail.load(&whole(0..2)), [12, 13, 16, 17]);
        assert_eq!([alias.addr(), tail.addr()], [head.addr() + 1, head.addr() + 2]);
    }

    /// Overlapping uses of one launch conflict when either writes, wherever each owner starts in
    /// the allocation; reads and other bytes do not.
    #[test]
    fn launch_uses_conflict_only_on_overlapping_writes() {
        let allocations = allocations();
        let buffer = Buffer::alloc(&allocations, Kind::DramShared, 4).unwrap();
        let alias = buffer.alias(2..).unwrap();
        let other = Buffer::alloc(&allocations, Kind::DramShared, 4).unwrap();
        let busy = |uses: &[(&Buffer, Region, Access)]| {
            let held = uses
                .iter()
                .map(|(buffer, region, access)| (buffer.placed(region).unwrap(), *access));
            matches!(rt::Lease::acquire(held), Err(rt::Error::Busy))
        };
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
            (&buffer, whole(0..4), Access::Write),
            (&other, whole(0..4), Access::Write)
        ]));
    }
}
