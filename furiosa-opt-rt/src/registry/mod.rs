//! Compiled `#[device]` functions the executable carries, and loading one onto a device.
//!
//! - The compiler's link pass writes an [`Entry`] per launched instantiation into section [`SECTION`], sorted
//!   by path then [`Key`], and each entry's [`image`] into its own section, so tools can read a binary's
//!   images without running it.
//! - The link pass lays entries out with its own build of this crate, so a program links the SDK its compiler
//!   was released with.

pub mod image;
mod load;

/// Spells the registry's section name once for the attributes and [`SECTION`].
macro_rules! section {
    () => {
        "furiosa_kernels"
    };
}

/// A compiled instantiation in the executable's registry, which is sorted by path and key.
///
/// The link pass writes each reference as its address followed by its length.
#[repr(C)]
pub struct Entry {
    /// The `#[device]` function's path.
    pub path: &'static str,
    /// The function's const-generic arguments in this instantiation.
    pub key: &'static [usize],
    /// The function-item type name for a type-keyed instantiation; empty for numeric keys.
    pub instance: &'static str,
    /// The compiler's image of this instantiation.
    pub image: &'static [u8],
}

/// What tells two instantiations of one function apart: the numeric values of its axes and
/// constants, or, for a function generic over other types, its function-item type name.
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

/// The registry the link pass wrote, sorted by path then key.
pub fn entries() -> &'static [Entry] {
    std::hint::black_box(&raw const ANCHOR);
    let start = &raw const START;
    let stop = &raw const STOP;
    let bytes = stop.addr() - start.addr();
    assert!(bytes.is_multiple_of(size_of::<Entry>()), "malformed native registry");
    // SAFETY: the bounds delimit the section's `Entry` values, which the link pass wrote in this crate's layout.
    unsafe { std::slice::from_raw_parts(start.cast::<Entry>(), bytes / size_of::<Entry>()) }
}

/// Where the entry of `path` with `key` sits in [`entries`]. The link pass assembles the registry after every
/// crate is compiled, so a function knows its unique `(path, key)` but not where its entry landed.
pub fn find(path: &str, key: Key<'_>) -> Option<usize> {
    entries()
        .binary_search_by(|entry| entry.path.cmp(path).then(entry.key().cmp(&key)))
        .ok()
}

/// The level a function's launches profile at. A level the image refuses goes to `refused` with
/// the reason, and the function loads unprofiled.
pub struct Profile {
    pub level: Level,
    pub refused: fn(Level, &crate::Error),
}

/// Profile levels retain the compiler's `ProfileLevel` and libpe's `tuc_profile_level` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    pub const ALL: [Self; 3] = [Self::Info, Self::Debug, Self::Trace];
}

/// The tracing target shared by decoded spans and the caller's listener check.
pub const TARGET: &str = "span::npu";

/// The section that holds the executable's [`Entry`] values, bounded by the linker's `__start_`/`__stop_` symbols.
pub const SECTION: &str = section!();

/// Entry `index`'s image sits in section `IMAGE_SECTION` followed by `index`.
pub const IMAGE_SECTION: &str = ".furiosa_image.";

unsafe extern "C" {
    #[link_name = concat!("__start_", section!())]
    static START: u8;
    #[link_name = concat!("__stop_", section!())]
    static STOP: u8;
}

// Defines the section and its bounds in a link the driver never saw (a plain cargo build), so
// that binary's registry is empty instead of failing to link.
#[used]
#[unsafe(link_section = section!())]
static ANCHOR: [Entry; 0] = [];
