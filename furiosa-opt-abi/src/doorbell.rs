//! A core's doorbells: a status word whose bits name the senders with pending entries, and one
//! submission ring per sender slot. Only where the registers lie differs between chips.

use core::ops::Range;

use crate::mmio::{Memory, Register};
use crate::ring::{Error, Ring};

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
        let (mapped, extent) = (memory.mapped(), layout.extent());
        (mapped.start <= extent.start && extent.end <= mapped.end)
            .then_some(Self { memory, layout })
            .ok_or(Error::Unmapped)
    }

    pub fn status(&self) -> u64 {
        // SAFETY: `new` checked the memory holds every register the layout places.
        unsafe { Register::<u64>::new(self.memory.origin().wrapping_add(self.layout.status).cast()) }.read_acquire()
    }

    /// The ring the sender at doorbell `slot` writes into.
    pub fn ring(&self, slot: u8) -> Result<Ring<M>, Error>
    where
        M: Clone,
    {
        if slot >= self.layout.slots {
            return Err(Error::Index);
        }
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
    pub slots: u8,
    pub stride: usize,
    /// Words in each slot's ring.
    pub capacity: usize,
}

impl Layout {
    /// The offsets the registers span, from the first ring's entries through the last register.
    pub fn extent(&self) -> Range<usize> {
        let slots = usize::from(self.slots);
        let rings = self.entries + slots * self.stride * size_of::<u64>();
        let indices = self.producer.max(self.consumer) + slots * size_of::<u32>();
        let status = self.status + size_of::<u64>();
        self.entries.min(self.producer).min(self.consumer).min(self.status)..rings.max(indices).max(status)
    }
}
