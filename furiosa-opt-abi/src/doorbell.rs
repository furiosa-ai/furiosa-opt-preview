//! A core's doorbells: a status word whose bits name the senders with pending entries, and one
//! submission ring per sender slot. Only where the registers lie differs between chips.

use crate::mmio::Register;
use crate::ring::{Error, Ring};

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

/// One core's doorbells: its address zero, and where its registers lie from there.
#[derive(Debug)]
pub struct Doorbells {
    base: *mut u8,
    layout: &'static Layout,
}

impl Doorbells {
    /// One core's doorbells, with that core's address zero at `base`: null for this core's own.
    ///
    /// # Safety
    ///
    /// `base` must map that core's doorbells, as `layout` places them, while the value is used.
    pub const unsafe fn at(base: *mut u8, layout: &'static Layout) -> Self {
        Self { base, layout }
    }

    pub fn status(&self) -> u64 {
        self.register(self.layout.status).read()
    }

    /// The ring the sender at doorbell `slot` writes into.
    pub fn ring(&self, slot: u8) -> Result<Ring, Error> {
        if slot >= self.layout.slots {
            return Err(Error::Index);
        }
        let slot = usize::from(slot);
        let entries = self.layout.entries + slot * self.layout.stride * size_of::<u64>();
        let index = slot * size_of::<u32>();
        // SAFETY: construction retains every ring and index a slot selects.
        Ok(unsafe {
            Ring::new(
                self.register(entries),
                self.register(self.layout.producer + index),
                self.register(self.layout.consumer + index),
                self.layout.capacity,
            )
        })
    }

    fn register<T: Copy>(&self, offset: usize) -> Register<T> {
        // SAFETY: construction maps every register the layout places from `base`.
        unsafe { Register::new(self.base.wrapping_add(offset).cast()) }
    }
}
