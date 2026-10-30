//! Tensors placed on memory, one file per placement.
//!
//! `host`, `hbm`, `spm` and `dm` each hold a placement's type, its views and the operations that
//! stay within it; `register_file` holds the two an engine keeps in registers. [`dma`] holds the
//! families that move data between placements, and `redistribute` the SRAM reshuffle composed
//! from them.

mod dm;
mod dma;
mod hbm;
mod hbm_scalar;
mod host;
mod redistribute;
mod register_file;
pub(crate) mod spm;

pub use dm::{DmAllocError, DmTensor, DmTensorView, DmTensorViewMut};
pub use dma::*;
pub use hbm::{AllChips, HbmTensor, HbmTensorView, HbmTensorViewMut, SomeChips};
pub use hbm_scalar::HbmScalar;
pub use host::HostTensor;
pub use redistribute::*;
pub use register_file::{TrfTensor, VrfTensor};
pub use spm::SpmTensor;

use dm::SliceIndexing;
