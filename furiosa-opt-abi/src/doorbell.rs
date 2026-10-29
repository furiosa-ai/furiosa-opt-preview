//! A core's doorbells: a status word whose bits name the senders with pending entries, and one
//! submission ring per sender slot. Only where the registers lie differs between chips.

use core::ops::Range;

use crate::mmio::{Memory, Register};
use crate::ring::{Error, Ring, disjoint, holds, span};

/// One core's doorbells: the memory they lie in, with that core's address zero as its origin, and
/// where its registers lie from there.
#[derive(Debug)]
pub struct Doorbells<M> {
    memory: M,
    layout: &'static Layout,
}

impl<M: Memory> Doorbells<M> {
    /// The doorbells `layout` places in `memory`, whose origin is the core's address zero.
    pub fn new(memory: M, layout: &'static Layout) -> Result<Self, Error> {
        let (mapped, extent) = (memory.mapped(), layout.extent()?);
        if !(mapped.start <= extent.start && extent.end <= mapped.end) {
            return Err(Error::Unmapped);
        }
        // Each ring checks its own registers when it is made.
        holds::<u64>(&memory, layout.status, 1)?;
        Ok(Self { memory, layout })
    }

    pub fn status(&self) -> u64 {
        // SAFETY: `new` checked the memory holds the status register, aligned.
        unsafe { Register::<u64>::new(self.memory.origin().wrapping_add(self.layout.status).cast()) }.read_acquire()
    }

    /// The ring the sender at doorbell `slot` writes into. A slot is one sender's, so that sender
    /// alone makes a [`Producer`](crate::ring::Producer) of it, once; a consumer may be made anew
    /// on every look, as it re-reads its index.
    pub fn ring(&self, slot: u8) -> Result<Ring<M>, Error>
    where
        M: Clone,
    {
        if slot >= self.layout.slots {
            return Err(Error::Index);
        }
        // `new` checked the last slot's offsets do not overflow.
        let slot = usize::from(slot);
        let entries = self.layout.entries + slot * self.layout.stride * size_of::<u64>();
        let index = slot * size_of::<u32>();
        Ring::new(
            self.memory.clone(),
            entries,
            self.layout.producer + index,
            self.layout.consumer + index,
            self.layout.capacity,
        )
    }
}

/// Where a core's doorbell registers lie, as byte offsets from that core's address zero.
#[derive(Debug)]
pub struct Layout {
    pub status: usize,
    /// The first slot's ring entries; slot `n`'s entries start `n * stride` words later.
    pub entries: usize,
    /// Arrays of one `u32` index per slot.
    pub producer: usize,
    pub consumer: usize,
    /// Senders, one bit each in the status word.
    pub slots: u8,
    pub stride: usize,
    /// Words in each slot's ring.
    pub capacity: usize,
}

impl Layout {
    /// The offsets the registers span, from the lowest through the end of the highest. Refuses a
    /// layout whose registers or slots' rings share a byte, whose slots outnumber the status bits,
    /// or whose offsets overflow.
    pub fn extent(&self) -> Result<Range<usize>, Error> {
        if u32::from(self.slots) > u64::BITS {
            return Err(Error::Slots);
        }
        if self.stride < self.capacity {
            return Err(Error::Overlap);
        }
        let slots = usize::from(self.slots);
        let rings = slots.checked_mul(self.stride).ok_or(Error::Unmapped)?;
        let regions = [
            span(self.status, 1, size_of::<u64>())?,
            span(self.entries, rings, size_of::<u64>())?,
            span(self.producer, slots, size_of::<u32>())?,
            span(self.consumer, slots, size_of::<u32>())?,
        ];
        disjoint(&regions)?;
        let start = regions.iter().map(|region| region.start).min();
        let end = regions.iter().map(|region| region.end).max();
        Ok(start.unwrap_or(0)..end.unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmio::Fixed;

    /// One slot's doorbells over five words: status first, the index arrays, then the ring.
    const LAYOUT: Layout = Layout {
        status: 0,
        entries: 24,
        producer: 16,
        consumer: 20,
        slots: 1,
        stride: 2,
        capacity: 2,
    };

    fn leak(layout: Layout) -> &'static Layout {
        Box::leak(Box::new(layout))
    }

    /// Doorbells whose status register is off its alignment are refused.
    #[test]
    fn refuses_misaligned_status() {
        let mut words = [0u64; 5];
        // SAFETY: `words` outlives every use of the memory.
        let memory = unsafe { Fixed::new(words.as_mut_ptr().cast(), 0..size_of_val(&words)) };
        assert!(Doorbells::new(memory, &LAYOUT).is_ok());
        assert_eq!(
            Doorbells::new(memory, leak(Layout { status: 4, ..LAYOUT })).err(),
            Some(Error::Misaligned)
        );
    }

    /// A layout whose registers or slots' rings share a byte, whose slots outnumber the status
    /// bits, or whose offsets overflow is refused.
    #[test]
    fn refuses_unsound_layouts() {
        let two = Layout {
            slots: 2,
            producer: 8,
            consumer: 16,
            ..LAYOUT
        };
        assert!(two.extent().is_ok());
        assert_eq!(Layout { stride: 1, ..two }.extent(), Err(Error::Overlap));
        assert_eq!(Layout { status: 16, ..LAYOUT }.extent(), Err(Error::Overlap));
        assert_eq!(Layout { consumer: 18, ..LAYOUT }.extent(), Err(Error::Overlap));
        assert_eq!(Layout { slots: 65, ..LAYOUT }.extent(), Err(Error::Slots));
        assert_eq!(
            Layout {
                stride: usize::MAX / 8,
                ..two
            }
            .extent(),
            Err(Error::Unmapped)
        );
    }
}
