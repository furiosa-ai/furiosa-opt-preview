//! Backends: [`Cpu`] interprets device functions on the host; [`Npu`] runs them on real chips.
//! - The cfg-selected [`crate::runtime::CurrentBackend`] alias picks the one the crate compiles
//!   against.

pub(crate) mod op_prep;

mod cpu;
/// Index units and validation for indirect DMA.
pub mod indirect;
/// NPU backend.
pub(crate) mod npu;

pub use cpu::Cpu;
pub use indirect::{IndexUnit, IndexValueError};
pub use npu::Npu;

pub(crate) mod internal;
pub(crate) use internal::{Allocation, Runtime};

/// Where device functions run: [`Cpu`] or [`Npu`]. Only these two implement it; what a backend
/// does with devices, memory and tensors is internal to the runtime.
pub trait Backend: internal::Backend {}

impl<B: internal::Backend> Backend for B {}
