/// Which device memory an allocation or a compiler static lives in. A DRAM allocation holds the
/// same offset on every chip of its device, so one address names it on each chip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub enum Kind {
    /// DRAM for its own chip's device code alone; the default. Nothing keeps peers out: it is
    /// placed outside the peer window while free spans allow, to leave that scarce DRAM to
    /// `DramShared`, and inside it otherwise.
    Dram,
    /// The SRAM every PE addresses at one fixed virtual address. A resident allocation lives until
    /// its last owner drops it; a launch's temporaries take the SRAM below its image's `sram_end`
    /// for the launch's duration. The host cannot transfer SRAM.
    Sram,
    /// DRAM inside the window every peer chip maps (see [`crate::arch::rngd::ATU_ENTRIES`]), so
    /// another chip's device code may write it. The window is a few 4 GiB entries, so this memory
    /// is scarce; a function's outputs and peer exchanges need it.
    DramShared,
}
