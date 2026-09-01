mod backend;
mod bind;
mod bindings;
mod ffi;
mod kernel;
mod output;
mod registry;

pub use backend::Npu;
pub use ffi::NpuDesc;
pub(crate) use ffi::bind_device;
pub(crate) use kernel::CpuBuffer;
pub use kernel::{Binding, Kernel, Kernels, kernel};
pub use registry::Key;

#[doc(hidden)]
pub mod __private {
    // TODO: Replace structural `Bindings` with `[Binding; N]` when stable Rust supports
    // `A::COUNT + B::COUNT` and `T::COUNT * N` in public return types without downstream gates.
    pub use super::bindings::{Bindings, empty, join, run, run_into};
    pub use super::output::DeviceOutput;
}
