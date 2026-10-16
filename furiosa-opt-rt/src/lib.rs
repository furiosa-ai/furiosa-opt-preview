//! Device execution, DMA, and resource lifetime.

mod arch {
    pub(crate) mod rngd;
}
mod buffer;
mod device;
mod driver;
mod function;
pub mod memory;
mod pinned;
mod placement;
pub mod registry;
#[cfg(test)]
mod test;

pub use buffer::{Access, Buffer, BufferSlice, BufferSliceMut, Lease};
pub use device::Device;
pub use driver::link::Sample;
pub use function::{Argument, Arguments, Function, Launch};
pub use pinned::{HostBufferSlice, HostBufferSliceMut, Pinned};
pub use placement::{Placement, ToPlacements, Topology};

/// Why the runtime refused or failed a device operation.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Another launch or transfer still uses the bytes; it is admitted once that work finishes.
    #[error("the buffer range is in use")]
    Busy,
    /// The memory was held by device work that failed, so its contents are unknown.
    #[error("the memory is unavailable after a device failure")]
    Poisoned,
    /// A launch bound more arguments than its function holds.
    #[error("more arguments than the function's signature holds")]
    Capacity,
    /// The caller asked for something the device or the program cannot do.
    #[error("invalid request: {0}")]
    Invalid(String),
    /// The hardware or its driver failed.
    #[error("device error: {0}")]
    Device(String),
    /// A device task panicked; carries its message so the caller can raise it as its own panic.
    #[error("a device task {0}")]
    Panicked(String),
    /// Another process holds the PEs this device needs on chip `npu{0}`.
    #[error("npu{0} is already reserved by another process")]
    Reserved(u8),
    /// The buffer belongs to another device.
    #[error("the buffer belongs to another device")]
    ForeignBuffer,
    /// No free span of `kind` memory holds `requested` bytes.
    #[error("{kind:?} has no contiguous space for {requested} bytes")]
    OutOfMemory { kind: memory::Kind, requested: usize },
    /// SRAM residents and a launch's temporaries would overlap: the lowest resident starts at
    /// `resident`, below where the temporaries end.
    #[error("SRAM residents from {resident} overlap launch temporaries up to {temporaries}")]
    SramOverlap { resident: usize, temporaries: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn reserved(chip: u8, error: nix::errno::Errno) -> Self {
        match error {
            nix::errno::Errno::EBUSY | nix::errno::Errno::EAGAIN => Self::Reserved(chip),
            error => Self::Device(format!("reserving npu{chip}: {error}")),
        }
    }
}

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme {}
