//! Device memory: its kinds, the layout an allocation asks for, the allocator that places it, and
//! how much of a kind is free.

pub use crate::buffer::Allocator;
pub use furiosa_opt_abi::memory::Kind;

/// The bytes an allocation holds on each chip, and a power of two its start is a multiple of; the
/// allocator may align it further. A size alone, `Layout::from(size)`, asks for no alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    size: usize,
    alignment: usize,
}

impl Layout {
    /// `size` bytes starting on a multiple of `alignment`.
    ///
    /// # Panics
    /// If `alignment` is not a power of two.
    pub fn new(size: usize, alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "alignment {alignment} is not a power of two"
        );
        Self { size, alignment }
    }

    /// Bytes on each chip.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The power of two the allocation's start is a multiple of.
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

impl From<usize> for Layout {
    fn from(size: usize) -> Self {
        Self::new(size, 1)
    }
}

/// Free bytes per chip of one memory kind; a snapshot that reserves nothing for later allocations.
/// It sizes a pool before allocating it, as a KV cache that takes what the device has left;
/// `largest` bounds one allocation, since free bytes need not be contiguous, and an allocation can
/// still fail with `Error::OutOfMemory` once other work allocates first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    /// Free bytes after subtracting live allocations and their alignment padding.
    pub available: usize,
    /// Largest contiguous free span; zero when exhausted.
    pub largest: usize,
}
