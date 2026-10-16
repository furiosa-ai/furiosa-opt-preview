//! Native compiler programs linked with the same Rust SDK and toolchain as the runtime.

use std::pin::Pin;
use std::sync::Arc;

use crate::{Device, Function, Result};

/// Loads its compiled program, linked into the executable, onto a device.
/// Loading may allocate; submission and completion use preallocated storage.
pub type Image = fn(Arc<Device>, Option<Profile>) -> Pin<Box<dyn Future<Output = Result<Function>> + Send>>;

/// The tracing target shared by decoded spans and the caller's listener check.
pub const TARGET: &str = "span::npu";

/// Capture levels retain the compiler's `ProfileLevel` and libpe's `tuc_profile_level` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    pub const ALL: [Self; 3] = [Self::Info, Self::Debug, Self::Trace];
}

/// An optional capture request; refusal is reported without failing the function load.
pub struct Profile {
    pub level: Level,
    pub report: fn(Level, &crate::Error),
}

/// A compiled instantiation in the executable's native registry, which is sorted by path and key.
///
/// The compiler's link pass emits these into the `furiosa_kernels` link section, and
/// `furiosa-opt-std` and the loader's registry test read the section back as a slice, so both
/// sides must be built by the same rustc and nothing else may be placed in that section.
#[repr(C)]
pub struct Entry {
    /// The `#[device]` function's path.
    pub path: &'static str,
    /// The function's const-generic arguments in this instantiation.
    pub key: &'static [usize],
    /// The function-item type name for a type-keyed instantiation; empty for numeric keys.
    pub instance: &'static str,
    pub image: Image,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[doc(hidden)]
pub enum Key<'a> {
    Numeric(&'a [usize]),
    Type(&'static str),
}

impl Entry {
    #[doc(hidden)]
    pub fn key(&self) -> Key<'_> {
        match self.instance {
            "" => Key::Numeric(self.key),
            instance => Key::Type(instance),
        }
    }
}
