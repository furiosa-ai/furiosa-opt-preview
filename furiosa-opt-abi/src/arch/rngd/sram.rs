//! Resident SRAM window bound by the host runtime.
//!
//! Compiler temporaries grow from zero; residents grow down from SRAM end.
//! Admission keeps temporaries below every live resident.

use core::ops::Range;

/// Physical bytes a resident may occupy: every page of a slice. What keeps a launch's temporaries
/// out of them is the admission check, not a page held back for them.
pub const WINDOW: Range<u64> = 0..16 * PAGE;

/// A byte of [`WINDOW`], by its physical offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Physical(u64);

impl TryFrom<u64> for Physical {
    /// The offset, as it came: it lies outside [`WINDOW`].
    type Error = u64;

    fn try_from(offset: u64) -> Result<Self, u64> {
        WINDOW.contains(&offset).then_some(Self(offset)).ok_or(offset)
    }
}

impl Physical {
    /// The window's first byte.
    pub const START: Self = Self(WINDOW.start);

    /// The address the kernel uses for this byte.
    pub const fn virtual_addr(self) -> u64 {
        self.0 - WINDOW.start + VIRTUAL
    }
}

/// Alignment of a resident's base address.
pub const ALIGNMENT: u64 = 256;

/// Bytes per SRAM page on one slice.
const PAGE: u64 = 32 * 1024;

/// Virtual byte the kernel uses to address the first resident byte: the window maps one-to-one
/// onto the top of the 128-page table.
const VIRTUAL: u64 = 128 * PAGE - (WINDOW.end - WINDOW.start);

#[cfg(test)]
mod tests {
    use super::*;

    /// The window maps onto the top of the kernel's 128-page table; no offset past it parses.
    #[test]
    fn maps_the_window_onto_the_table_top() {
        assert_eq!(
            Physical::try_from(WINDOW.end - 1).map(Physical::virtual_addr),
            Ok(128 * PAGE - 1)
        );
        assert_eq!(Physical::try_from(WINDOW.end), Err(WINDOW.end));
    }
}
