//! The registry a binary carries: an [`Entry`] per launched `#[device]` instantiation, which the
//! compiler's link pass writes and the runtime reads.
//!
//! - The link pass writes the entries into section [`SECTION`], sorted by path then key, and each
//!   entry's image into its own section, so tools can read a binary's images without running it.
//! - The link pass lays entries out with its own build of this crate, so a program links the SDK its
//!   compiler was released with.

use core::mem::offset_of;

/// Spells the registry's section name once, for link attributes and [`SECTION`].
#[macro_export]
macro_rules! registry_section {
    () => {
        "furiosa_kernels"
    };
}

/// The section that holds a binary's [`Entry`] values, bounded by the linker's `__start_` and
/// `__stop_` symbols.
pub const SECTION: &str = registry_section!();

/// Entry `index`'s image sits in section `IMAGE_SECTION` followed by `index`.
pub const IMAGE_SECTION: &str = ".furiosa_image.";

/// A compiled instantiation in a binary's registry. Only the link pass makes entries: it writes a
/// [`Ref`] at each of [`Entry::PATH`], [`Entry::KEY`] and [`Entry::IMAGE`].
#[repr(C)]
pub struct Entry {
    path: Ref,
    key: Ref,
    image: Ref,
}

// SAFETY: an entry refers only to static data, which nothing writes.
unsafe impl Sync for Entry {}

impl Entry {
    /// Where an entry holds the reference to its path.
    pub const PATH: usize = offset_of!(Self, path);
    /// Where an entry holds the reference to its key.
    pub const KEY: usize = offset_of!(Self, key);
    /// Where an entry holds the reference to its image.
    pub const IMAGE: usize = offset_of!(Self, image);

    /// The `#[device]` function's path.
    pub fn path(&self) -> &'static str {
        // SAFETY: the link pass writes a path as its UTF-8 bytes.
        unsafe { core::str::from_utf8_unchecked(self.path.slice()) }
    }

    /// The function's const-generic arguments in this instantiation.
    pub fn key(&self) -> &'static [usize] {
        self.key.slice()
    }

    /// The compiler's image of this instantiation.
    pub fn image(&self) -> &'static [u8] {
        self.image.slice()
    }
}

/// A reference as the link pass writes it: the address of `len` values, which a relocation fills
/// in, then `len`.
#[repr(C)]
pub struct Ref {
    addr: *const u8,
    len: usize,
}

impl Ref {
    /// Where a reference holds its address.
    pub const ADDR: usize = offset_of!(Self, addr);
    /// Where a reference holds its length.
    pub const LEN: usize = offset_of!(Self, len);

    fn slice<T>(&self) -> &'static [T] {
        // SAFETY: a reference exists only inside an entry the link pass wrote, which points it at
        // `len` aligned values of its field's type in static data.
        unsafe { core::slice::from_raw_parts(self.addr.cast(), self.len) }
    }
}
