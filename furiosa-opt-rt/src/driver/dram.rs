use std::fs::File;
use std::ops::Range;
use std::os::fd::AsRawFd;

use super::kernel::{self, node};
use crate::arch::rngd;
use crate::{Error, Result};

/// One chip's memory window and the share of it the device's PEs own, an eighth per PE: the
/// firmware image at the share's top, everything below it for the allocator.
pub(crate) struct Dram {
    pub(crate) chip: u8,
    /// Where the firmware image is staged in the window.
    pub(crate) offset: usize,
    /// The offsets of the window the allocator may hand out: the share below the staged image and
    /// the firmware's DMA descriptors.
    pub(crate) free: Range<usize>,
    /// Whether the chip's first boot wrote the firmware image at `offset`.
    pub(crate) staged: bool,
}

impl Dram {
    /// The share of `chip`'s memory the PE bitmask `pes` owns, with `image` placed at its top;
    /// The returned placement leaves space for firmware DMA descriptors.
    pub(crate) fn open(chip: u8, pes: u8, image: usize) -> Result<Self> {
        let bar = File::options()
            .read(true)
            .write(true)
            .open(node(chip, "bar4"))
            .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "bar4"))))?;
        let mut usable = kernel::BarInfo::default();
        // SAFETY: `usable` is a valid writable `BarInfo` for this ioctl.
        unsafe { kernel::available_dram(bar.as_raw_fd(), &mut usable) }
            .map_err(|why| Error::Device(format!("asking for the usable window: {why}")))?;
        let chip_pes = usize::from(rngd::PES);
        // An even share of the usable DRAM per PE, in whole pages.
        let per_pe = (usable.size as usize / chip_pes) & !(PAGE - 1);
        let (first, past) = (pes.trailing_zeros() as usize, chip_pes - pes.leading_zeros() as usize);
        let share = first * per_pe..past * per_pe;
        let offset = share
            .end
            .saturating_sub(PAGE)
            .checked_sub(image)
            .filter(|offset| *offset >= share.start + rngd::PDMA_DESCRIPTORS)
            .ok_or_else(|| Error::Device(format!("{} bytes do not fit above the share's floor", image)))?
            & !(PAGE - 1);
        Ok(Self {
            chip,
            offset,
            free: share.start..offset - rngd::PDMA_DESCRIPTORS,
            staged: false,
        })
    }
}

pub(crate) const PAGE: usize = 4096;
