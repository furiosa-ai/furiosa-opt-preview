//! Device memory: its kinds, the allocator that places it, and how much of a kind is free.

pub use crate::buffer::Allocator;
pub use furiosa_opt_abi::memory::Kind;

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
