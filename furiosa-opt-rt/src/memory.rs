//! Device memory: its kinds, the allocators that place it, and how much of a kind is free.

pub use crate::buffer::Allocations;
pub use furiosa_opt_abi::memory::Kind;

/// Free bytes per chip of one memory kind; a snapshot that reserves nothing for later allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    /// Free bytes after subtracting live allocations and their alignment padding.
    pub available: usize,
    /// Largest contiguous free span; zero when exhausted.
    pub largest: usize,
}
