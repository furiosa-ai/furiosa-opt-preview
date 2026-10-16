//! Resident SRAM window bound by the host runtime.
//!
//! Compiler temporaries grow from zero; residents grow down from SRAM end.
//! Admission keeps temporaries below every live resident.

use core::ops::Range;

/// Bytes per SRAM page on one slice.
const PAGE: u64 = 32 * 1024;

/// Physical bytes a resident may occupy: every page of a slice. What keeps a launch's temporaries
/// out of them is the admission check, not a page held back for them.
pub const WINDOW: Range<u64> = 0..16 * PAGE;

/// Virtual byte the kernel uses to address the first resident byte: the window maps one-to-one
/// onto the top of the 128-page table.
const VIRTUAL: u64 = 128 * PAGE - (WINDOW.end - WINDOW.start);

/// Alignment of a resident's base address.
pub const ALIGNMENT: u64 = 256;

/// Virtual address of a byte in the resident physical window.
pub const fn virtual_addr(physical: u64) -> u64 {
    assert!(physical < WINDOW.end);
    physical - WINDOW.start + VIRTUAL
}
