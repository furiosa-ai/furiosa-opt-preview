//! Backends: [`Cpu`] interprets device functions on the host; [`Npu`] runs them on real chips.
//! - The cfg-selected [`crate::runtime::CurrentBackend`] alias picks the one the crate compiles
//!   against.

pub(crate) mod op_prep;

mod cpu;
/// Index units and validation for indirect DMA.
pub mod indirect;
/// NPU backend.
pub(crate) mod npu;

use furiosa_mapping::Mapping;

pub use cpu::Cpu;
pub use indirect::{IndexUnit, IndexValueError};
pub use npu::Npu;

/// Runtime prefix in the sparse axis's original logical coordinates.
#[derive(Debug)]
pub struct GatherPrefix<'a, Storage> {
    /// Live positions for each batch.
    pub lengths: &'a Storage,
    /// Layout of the lengths.
    pub batch: Mapping,
    /// Original sparse coordinates.
    pub sparse: Mapping,
    /// Beginning of the requested window.
    pub offset: i32,
}

/// Index coordinates and live limits for a gather.
#[derive(Debug)]
pub struct GatherRead<'a, Storage> {
    /// Entry coordinates of the index list.
    pub domain: &'a Mapping,
    /// Unit of each index value.
    pub unit: IndexUnit,
    /// Entries gathered per chip; `None` reads the whole list.
    pub prefix: Option<usize>,
    /// Runtime prefix for each batch.
    pub batch_prefix: Option<GatherPrefix<'a, Storage>>,
}

pub(crate) mod internal;
pub(crate) use internal::Allocation;
/// Named by [`crate::runtime::DeviceSend::prepare`]'s device parameter.
#[doc(hidden)]
pub use internal::Runtime;

/// Where device functions run: [`Cpu`] or [`Npu`]. Only these two implement it; what a backend
/// does with devices, memory and tensors is internal to the runtime.
pub trait Backend: internal::Backend {}

impl<B: internal::Backend> Backend for B {}
