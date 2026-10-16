/// Which device memory a static lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Kind {
    Dram,
    Sram,
    /// DRAM within every peer's addressable aperture.
    DramShared,
}
