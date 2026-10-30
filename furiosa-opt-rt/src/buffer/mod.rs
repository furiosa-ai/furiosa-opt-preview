//! Owned device regions and borrowed read/write views.

use std::ops::{Bound, RangeBounds};
use std::sync::{Arc, Mutex};

pub(crate) mod access;
mod allocator;
pub(crate) use access::{Active, Admission, Owner, Request, Waiters};
pub use access::{Blocks, Hold, Lease};
pub use allocator::Allocator;
pub(crate) use allocator::{Ranges, Reservation};

use crate::memory::Kind;
use crate::{Error, Result, Topology};

/// Owns one byte range at the same offset on every chip. Splitting transfers disjoint ownership
/// without copying data.
pub struct Buffer {
    offset: usize,
    len: usize,
    allocation: Arc<Allocation>,
}

impl Buffer {
    /// Borrows `bytes` of the range on every chip; [`Error::Invalid`] outside the buffer.
    pub fn slice(&self, bytes: impl RangeBounds<usize>) -> Result<BufferSlice<'_>> {
        let (offset, len) = bounds(bytes, self.len)?;
        Ok(BufferSlice {
            buffer: self,
            offset,
            len,
            chips: ((1u16 << self.topology().chips) - 1) as u8,
        })
    }

    /// Exclusively borrows `bytes` of the range on every chip; [`Error::Invalid`] outside the buffer.
    pub fn slice_mut(&mut self, bytes: impl RangeBounds<usize>) -> Result<BufferSliceMut<'_>> {
        self.slice(bytes).map(BufferSliceMut)
    }

    /// Divides the buffer into the owners of the bytes before and from `offset`.
    pub fn split_at(mut self, offset: usize) -> Result<(Self, Self)> {
        bounds(offset.., self.len)?;
        let tail = Self {
            offset: self.offset + offset,
            len: self.len - offset,
            allocation: Arc::clone(&self.allocation),
        };
        self.len = offset;
        Ok((self, tail))
    }

    /// Another owner of `bytes` of this buffer's range on every chip, sharing its allocation. A
    /// launch or transfer that conflicts with another through either owner is
    /// [`crate::Error::Busy`].
    pub fn alias(&self, bytes: impl RangeBounds<usize>) -> Result<Self> {
        Ok(self.slice(bytes)?.retain())
    }

    /// Extends this range by an adjacent range of the same allocation; aliases and gaps stay separate.
    pub fn join(&mut self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.allocation, &other.allocation) && self.offset + self.len == other.offset {
            self.len += other.len;
            true
        } else {
            false
        }
    }

    /// Whether `allocations` placed this buffer; a device refuses any other as
    /// [`Error::ForeignBuffer`].
    pub fn belongs_to(&self, allocator: &Arc<Allocator>) -> bool {
        Arc::ptr_eq(&self.allocation.allocator, allocator)
    }

    /// The memory kind the buffer was allocated from.
    pub fn kind(&self) -> Kind {
        self.allocation.kind
    }

    /// The chips the buffer spans, one range on each.
    pub fn topology(&self) -> Topology {
        self.allocation.allocator.topology()
    }

    /// Bytes the buffer owns on each chip.
    pub fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Where the buffer starts on each chip: an offset into the chip's DRAM, or the address device
    /// code reads SRAM at.
    pub fn addr(&self) -> usize {
        let base = match self.kind() {
            Kind::Dram | Kind::DramShared => 0,
            Kind::Sram => self.allocation.allocator.sram_base(),
        };
        base + self.allocation.at + self.offset
    }

    /// The number of the allocation the buffer is part of among its device's: bytes at an address
    /// a dropped allocation used belong to a later number.
    pub(crate) fn generation(&self) -> u64 {
        self.allocation.generation
    }

    /// The operand word device code addresses the buffer by, on a chip whose DRAM its clusters
    /// see at `window`.
    pub(crate) fn operand(&self, window: u64) -> u64 {
        match self.kind() {
            Kind::Dram | Kind::DramShared => window + self.addr() as u64,
            Kind::Sram => self.addr() as u64,
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("at", &self.addr())
            .field("len", &self.len)
            .finish()
    }
}

/// A shared device borrow of the same bytes on each selected chip.
#[derive(Clone, Copy, Debug)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    offset: usize,
    len: usize,
    chips: u8,
}

impl<'a> BufferSlice<'a> {
    /// The `bytes` of this view on the same chips; [`Error::Invalid`] outside the view.
    pub fn slice(mut self, bytes: impl RangeBounds<usize>) -> Result<Self> {
        let (offset, len) = bounds(bytes, self.len)?;
        self.offset += offset;
        self.len = len;
        Ok(self)
    }

    /// The view on chip `chip` alone; [`Error::Invalid`] unless the view selects it.
    pub fn on(mut self, chip: usize) -> Result<Self> {
        self.chips = u32::try_from(chip)
            .ok()
            .and_then(|chip| 1u8.checked_shl(chip))
            .filter(|bit| self.chips & bit != 0)
            .ok_or_else(|| Error::Invalid(format!("chip {chip} is not among the view's chips")))?;
        Ok(self)
    }

    /// The view on each selected chip alone, in rank order.
    pub fn chips(self) -> impl Iterator<Item = Self> {
        (0..usize::from(self.buffer.topology().chips)).filter_map(move |chip| self.on(chip).ok())
    }

    /// Bytes the view covers on each selected chip.
    pub fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Where the view starts on each chip, as [`Buffer::addr`] counts.
    pub fn addr(&self) -> usize {
        self.buffer.addr() + self.offset
    }

    /// The memory kind of the buffer the view borrows.
    pub fn kind(&self) -> Kind {
        self.buffer.kind()
    }

    /// The buffer the view borrows.
    pub(crate) fn buffer(&self) -> &'a Buffer {
        self.buffer
    }

    /// The chips the view selects, one bit per rank.
    pub(crate) fn chip_mask(&self) -> u8 {
        self.chips
    }

    /// The view on the chips of `chips`, one bit per rank, which the buffer must span.
    pub(crate) fn selecting(mut self, chips: u8) -> Self {
        self.chips = chips;
        self
    }

    /// Compares allocation, starting offset, and selected chips; extents may differ.
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer.allocation, &other.buffer.allocation)
            && self.addr() == other.addr()
            && self.chips == other.chips
    }

    pub(crate) fn retain(&self) -> Buffer {
        Buffer {
            offset: self.buffer.offset + self.offset,
            len: self.len,
            allocation: Arc::clone(&self.buffer.allocation),
        }
    }
}

/// An exclusive device borrow of the same bytes on each selected chip.
#[derive(Debug)]
pub struct BufferSliceMut<'a>(pub(crate) BufferSlice<'a>);

impl<'a> BufferSliceMut<'a> {
    /// The `bytes` of this view on the same chips; [`Error::Invalid`] outside the view.
    pub fn slice(self, bytes: impl RangeBounds<usize>) -> Result<Self> {
        self.0.slice(bytes).map(Self)
    }

    /// The view on chip `chip` alone; [`Error::Invalid`] unless the view selects it.
    pub fn on(self, chip: usize) -> Result<Self> {
        self.0.on(chip).map(Self)
    }

    /// The view on each selected chip alone, in rank order.
    pub fn chips(self) -> impl Iterator<Item = Self> {
        self.0.chips().map(Self)
    }
}

/// Whether device work may write memory or only read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// The operation only reads the memory.
    Read,
    /// The operation may write the memory.
    Write,
}

fn bounds(bytes: impl RangeBounds<usize>, size: usize) -> Result<(usize, usize)> {
    let start = match bytes.start_bound() {
        Bound::Unbounded => Some(0),
        Bound::Included(&value) => Some(value),
        Bound::Excluded(&value) => value.checked_add(1),
    };
    let end = match bytes.end_bound() {
        Bound::Unbounded => Some(size),
        Bound::Excluded(&value) => Some(value),
        Bound::Included(&value) => value.checked_add(1),
    };
    match (start, end) {
        (Some(start), Some(end)) if start <= end && end <= size => Ok((start, end - start)),
        _ => Err(Error::Invalid("buffer range is outside its owner".into())),
    }
}

struct Allocation {
    at: usize,
    /// This allocation's number among its device's, which tells it from another at the same
    /// address.
    generation: u64,
    kind: Kind,
    allocator: Arc<Allocator>,
    active: Mutex<access::Active>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.allocator.release(self.kind, self.at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slices, splits and chip selections stay inside their buffer.
    #[test]
    fn slices_stay_inside_their_buffer() {
        let outside = Some(Error::Invalid("buffer range is outside its owner".into()));
        let allocator = crate::test::allocator(0..1024, 1024);
        let buffer = allocator.alloc(Kind::Dram, 256).unwrap();
        assert_eq!(buffer.slice(0..257).err(), outside);
        assert_eq!(buffer.slice((Bound::Included(1), Bound::Excluded(0))).err(), outside);
        assert_eq!(buffer.slice(..=usize::MAX).err(), outside);
        let start = buffer.addr();
        let left = buffer.slice(..128).unwrap();
        assert_eq!((left.addr(), left.len()), (start, 128));
        assert_eq!(left.slice(127..129).err(), outside);
        assert_eq!(left.slice(128..).unwrap().len(), 0);
        assert_eq!(
            buffer.slice(..).unwrap().on(1).err(),
            Some(Error::Invalid("chip 1 is not among the view's chips".into()))
        );
        let (head, tail) = buffer.split_at(128).unwrap();
        assert_eq!((tail.addr(), tail.len()), (head.addr() + 128, 128));
        assert_eq!(tail.split_at(129).err(), outside);
    }
}
