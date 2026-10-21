//! Device executable and hardware contracts.

#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "bincode")]
extern crate bincode2 as bincode;
extern crate thiserror_core as thiserror;

pub mod bootloader;
pub mod doorbell;
pub mod executable;
pub mod memory;
pub mod mmio;
pub mod ring;

/// Chip-specific layouts and limits.
pub mod arch {
    pub mod rngd;
}

/// Number of chips and processing elements per chip in a compiled program or opened device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Topology {
    pub chips: u8,
    pub pes: u8,
}

impl From<(u8, u8)> for Topology {
    fn from((chips, pes): (u8, u8)) -> Self {
        Self { chips, pes }
    }
}

impl Topology {
    /// Clusters across all chips, each fusing at most [`CLUSTER_PES`](arch::rngd::CLUSTER_PES) PEs.
    pub const fn clusters(self) -> usize {
        self.chips as usize * self.pes.div_ceil(arch::rngd::CLUSTER_PES) as usize
    }
}
