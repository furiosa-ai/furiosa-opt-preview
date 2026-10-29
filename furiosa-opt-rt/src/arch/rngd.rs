//! RNGD's hardware addressing as the runtime uses it: where the firmware sees memory, the PCIe
//! translation entries and the PDMA engine.

pub(crate) use furiosa_opt_abi::arch::rngd::*;

/// Where the runtime's share of a chip's DRAM starts in the chip's DRAM, past the region the SoC
/// and PE firmware own (the kernel driver's `NPU_FW_SIZE`). The runtime points the PEs' memory
/// window here (`driver::cluster`), so a PE sees it at `DEVICE_MEMORY.start`, and the driver's
/// usable BAR4 range starts here, [`kernel::BAR4_DRAM`](crate::driver::kernel::BAR4_DRAM) into the
/// BAR. A byte the host writes at offset `x` is therefore at `DEVICE_MEMORY.start + x` for a PE.
pub(crate) const DRAM_FIRMWARE: u64 = 512 << 20;

/// Where the firmware sees the PDMA registers through the first translation entry.
pub(crate) const PDMA_REGISTERS: u64 = 0x80_4000_0000;
/// RNGD PDMA BAR0 register bank, including the driver's reserved register page.
pub(crate) const PDMA_EXTENT: u64 = (PDMA_REGISTERS - 0x80_0000_0000) + 0x1000;
/// RNGD PDMA reserves eight 8 KiB descriptor chains, one per physical channel.
pub(crate) const PDMA_DESCRIPTORS: usize = 8 * (8 << 10);
/// Where the firmware sees its own chip's memory for PDMA. The engine addresses DRAM as BAR4 maps
/// it, from the kernel driver's `NPU_DRAM_INWARD_ADDR`, and a PE's window starts where BAR4's
/// usable DRAM does, [`BAR4_DRAM`](crate::driver::kernel::BAR4_DRAM) in.
pub(crate) const LOCAL_DRAM: u64 = 0xc0_0000_0000 + crate::driver::kernel::BAR4_DRAM;
