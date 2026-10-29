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

/// The PCIe outbound translation entries through which a chip reaches its peers' DRAM, each a
/// window of [`ATU_ENTRY`] bytes. A device of `n` chips maps each chip's own memory in its first `n`
/// entries and gives member `rank` the `(ATU_ENTRIES - n) / n` consecutive entries from
/// `n + rank * (ATU_ENTRIES - n) / n`; for `n = 4`, entries 4-6, 7-9, 10-12 and 13-15.
///
/// The windows of one member must be consecutive: the PE DMA path restores only the low 32 bits of
/// a destination, so a window base must advance by exactly 4 GiB, and any other layout redirects
/// transfers without an error. The kernel driver (`__set_remote_entry`), the task library
/// (`get_base_raw_dram`) and the PE address map compute the same layout; this is normative for the
/// code the compiler emits, and a change to one is a change to all.
pub const ATU_ENTRIES: usize = 16;
/// Bytes one translation entry maps.
pub const ATU_ENTRY: usize = 4 << 30;

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
