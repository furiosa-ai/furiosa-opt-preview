//! Virtual ISA standard library.

#![expect(incomplete_features)]
#![feature(adt_const_params)]
#![feature(float_erf)]
#![feature(inherent_associated_types)]
#![feature(impl_trait_in_assoc_type)]
#![feature(once_cell_try)]
#![feature(slice_range)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![forbid(unused_must_use)]
#![expect(clippy::type_complexity)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#![allow(clippy::disallowed_types)]

mod array_vec;
mod cast;
mod constraints;
mod context;
mod control;
mod engine;
mod float;
mod scalar;
mod storage;
mod tensor;

pub mod backend;
pub mod runtime;

/// Internal APIs used by generated code.
#[doc(hidden)]
pub mod __private {
    use crate::backend::Runtime;
    pub use crate::control::__loop_hint_unroll;
    use crate::runtime::{CurrentBackend, DeviceOutput, DeviceSend, Kernel, Launched, Returns, Submitted};

    /// Runs `kernel` with `args` on the selected backend, writing its returns where `returns`
    /// says; `body` is the function's Rust body, which the host backend runs.
    #[allow(clippy::unit_arg, reason = "the host backend's kernel is `()`")]
    pub fn launch<F, Inputs, Args: DeviceSend, Output: DeviceOutput, R: Returns<Output>>(
        device: &mut crate::Device,
        kernel: Result<Kernel<F, Inputs>, crate::Error>,
        args: Args,
        returns: R,
        body: impl FnOnce(&mut crate::Device, Args) -> Output,
    ) -> Submitted<Launched<R::Value>> {
        returns.launch(device, |device, output| match kernel {
            Ok(kernel) => CurrentBackend::launch(device, kernel.id, args, output, body),
            Err(error) => Submitted::launched(Err(error)),
        })
    }
}

pub use runtime::{Device, HbmBuffer, Placement, ToPlacements, Topology};

/// Why a device, a transfer or a launch failed: the runtime refused or failed the work, or a
/// tensor was not where the operation needs it.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The runtime refused or failed an open, a load, a launch or a transfer.
    #[error(transparent)]
    Device(#[from] furiosa_opt_rt::Error),
    /// A `DmTensor` passed to a launch holds no resident SRAM (`DmTensor::alloc`).
    #[error("the DM tensor holds no resident SRAM")]
    Unplaced,
    /// The caller asked for a range, chips, length or layout the memory cannot take.
    #[error("invalid request: {0}")]
    Invalid(&'static str),
    /// The binary carries no compiled image for this device function and generic arguments.
    #[error("no compiled device function for `{path}` with generic arguments {key:?}")]
    Uncompiled {
        /// The function's registry path.
        path: &'static str,
        /// Its generic arguments' values.
        key: Vec<usize>,
    },
    /// The binary carries no compiled image for this function-item type.
    #[error("no compiled device function for `{path}` with instance `{instance}`")]
    UncompiledInstance {
        /// The function's registry path.
        path: &'static str,
        /// Rust's monomorphized function-item type name.
        instance: &'static str,
    },
    /// An earlier panic left the runtime's state unusable.
    #[error("the runtime is poisoned by an earlier panic")]
    Poisoned,
}

impl From<Error> for std::io::Error {
    fn from(error: Error) -> Self {
        let kind = match &error {
            Error::Device(furiosa_opt_rt::Error::OutOfMemory { .. }) => std::io::ErrorKind::OutOfMemory,
            _ => std::io::ErrorKind::Other,
        };
        Self::new(kind, error)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn distinguishes_memory_exhaustion() {
        use furiosa_opt_rt::{Error, memory::Kind};

        for (error, kind) in [
            (
                Error::OutOfMemory {
                    kind: Kind::Dram,
                    requested: 4096,
                },
                std::io::ErrorKind::OutOfMemory,
            ),
            (Error::Poisoned, std::io::ErrorKind::Other),
        ] {
            let error = std::io::Error::from(super::Error::Device(error));
            assert_eq!(error.kind(), kind);
            assert!(error.get_ref().unwrap().is::<super::Error>());
        }
    }
}

pub use tensor::pseudo;

/// Prelude module that re-exports commonly used items.
pub mod prelude {
    pub use super::backend::{Backend, Cpu, IndexUnit, IndexValueError, Npu};
    pub use super::cast::{
        Cast, CastEngineCast, CommitCast, ContractionAccumulator, ContractionCast, ContractionWeight, FetchCast,
        FetchZeroPointSub, TableLookupCast,
    };
    pub use super::context::{Dma, DmaContext, Tu, TuContext};
    pub use super::engine::vector::stash_slot::*;
    pub use super::engine::vector::{alu::*, branch::*, layer::*, op::*, operand::*, scalar::*, stage, tensor::*};
    pub use super::engine::*;
    pub use super::runtime::{
        Access, Bind, Chips, CurrentBackend, Device, DeviceError, DeviceFn, DeviceSend, HbmBuffer, HbmBufferSlice,
        HbmBufferSliceMut, HostBuffer, HostBufferSlice, HostBufferSliceMut, Kernel, Loadable, Placement, ToPlacements,
        Topology, launch,
    };
    pub use super::tensor::memory::*;
    pub use super::tensor::tu::*;
    pub use super::{Error, array_vec::*, constraints::SLICES_PER_PE, scalar::*, storage::*, tensor::*};
    pub use furiosa_mapping::*;
    pub use furiosa_opt_macro::{DeviceSend, device, unroll};
}
