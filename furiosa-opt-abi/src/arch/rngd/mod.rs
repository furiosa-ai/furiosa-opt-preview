//! RNGD address layout and execution limits.

use crate::doorbell::Layout;
use crate::mmio::Fixed;
use crate::ring::CompletionPage;
use core::mem::offset_of;
use core::ops::Range;

pub mod dma;
pub mod sram;

/// Chips one device spans at most.
pub const MAX_CHIPS: u8 = 8;
/// PEs on one chip.
pub const PES: u8 = 8;
/// DRAM bytes per chip the RNGD driver offers for allocation, split evenly among the PEs.
pub const DRAM: u64 = 51_002_736_640;
/// PEs one cluster fuses at most.
pub const CLUSTER_PES: u8 = 4;
/// Frequency programmed into CNTFRQ_EL0 by the bootloader, in ticks per second.
pub const COUNTER_HZ: u64 = 25_000_000;

/// Scratchpad region reserved for task instructions by the firmware linker.
pub const CODE: Range<usize> = 0x10_0000..0x31_0000;

/// The device's DRAM aperture in a PE's address space, per the RNGD memory map.
pub const DEVICE_MEMORY: Range<u64> = 0xc0_0000_0000..0xd0_0000_0000;

/// Doorbell registers in a PE's MMIO space, per the RNGD register map.
pub const DOORBELLS: Layout = Layout {
    status: 0x80_0cd8,
    entries: 0x40_0000,
    producer: 0x80_0800,
    consumer: 0x80_0900,
    slots: 64,
    stride: 128,
    capacity: 127,
};

/// Words in a completion ring.
pub const COMPLETION_RING: usize = 255;

/// The page every core's completion ring lies in.
// SAFETY: the standalone memory map places the completion page here in every core.
pub const COMPLETION: Fixed = unsafe {
    Fixed::new(
        core::ptr::without_provenance_mut(0x50_0020_0000),
        0..offset_of!(CompletionPage, entries) + COMPLETION_RING * size_of::<u64>(),
    )
};
