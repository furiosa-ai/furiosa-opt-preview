//! RNGD's hardware addressing as the runtime uses it: where the firmware sees memory, the PCIe
//! translation entries and the PDMA engine.

pub(crate) use furiosa_opt_abi::arch::rngd::*;

/// Device memory the firmware's PEs address, past the driver's own reservation.
pub(crate) const DRAM_FIRMWARE: u64 = 512 << 20;

/// Where a chip's memory window may start in a cluster's address space; the bootloader tells
/// which by finding the firmware image at it.
pub(crate) const CANDIDATES: [u64; 3] = [0, 256 << 20, 512 << 20];

/// The PCIe outbound translation entries one chip has, each mapping this many bytes.
pub(crate) const ATU_ENTRIES: usize = 16;
pub(crate) const ATU_ENTRY: usize = 4 << 30;

/// Where the firmware sees the PDMA registers through the first translation entry.
pub(crate) const PDMA_REGISTERS: u64 = 0x80_4000_0000;
/// RNGD PDMA BAR0 register bank, including the driver's reserved register page.
pub(crate) const PDMA_EXTENT: u64 = (PDMA_REGISTERS - 0x80_0000_0000) + 0x1000;
/// RNGD PDMA reserves eight 8 KiB descriptor chains, one per physical channel.
pub(crate) const PDMA_DESCRIPTORS: usize = 8 * (8 << 10);
/// Where the firmware sees its own chip's memory for PDMA.
pub(crate) const LOCAL_DRAM: u64 = 0xc0_1000_0000;
