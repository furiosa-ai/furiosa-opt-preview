//! Compiled `#[device]` functions the executable carries, and loading one onto a device.
//!
//! - The compiler's link pass writes the registry and its images in the layout and format
//!   `furiosa_opt_abi::registry` and `furiosa_opt_abi::image` define; [`entries`] reads it back.

mod load;
pub use furiosa_opt_abi::image::Level;
pub use furiosa_opt_abi::registry::{Entry, Key};
pub use load::load;

/// The registry the link pass wrote, sorted by path then key.
pub fn entries() -> &'static [Entry] {
    std::hint::black_box(&raw const ANCHOR);
    let start = &raw const START;
    let stop = &raw const STOP;
    let bytes = stop.addr() - start.addr();
    assert!(bytes.is_multiple_of(size_of::<Entry>()), "malformed native registry");
    // SAFETY: the bounds delimit the section's `Entry` values, which the link pass wrote in the
    // layout `Entry` defines.
    unsafe { std::slice::from_raw_parts(start.cast::<Entry>(), bytes / size_of::<Entry>()) }
}

/// Where the entry of `path` with `key` sits in [`entries`]. The link pass assembles the registry after every
/// crate is compiled, so a function knows its unique `(path, key)` but not where its entry landed.
pub fn find(path: &str, key: Key<'_>) -> Option<usize> {
    entries()
        .binary_search_by(|entry| entry.path().cmp(path).then(entry.key().cmp(&key)))
        .ok()
}

/// The level a function's launches profile at. A level the image refuses goes to `refused` with
/// the reason, and the function loads unprofiled.
pub struct Profile {
    pub level: Level,
    pub refused: fn(Level, &crate::Error),
}

/// The tracing target shared by decoded spans and the caller's listener check.
pub const TARGET: &str = "span::npu";

unsafe extern "C" {
    #[link_name = concat!("__start_", furiosa_opt_abi::registry_section!())]
    static START: u8;
    #[link_name = concat!("__stop_", furiosa_opt_abi::registry_section!())]
    static STOP: u8;
}

// Defines the section and its bounds in a link the driver never saw (a plain cargo build), so
// that binary's registry is empty instead of failing to link.
#[used]
#[unsafe(link_section = furiosa_opt_abi::registry_section!())]
static ANCHOR: [Entry; 0] = [];
