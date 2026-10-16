use std::ops::Range;
use std::os::fd::AsRawFd;

use super::map::open_rw;
use crate::arch::rngd::{self, node};
use crate::{Error, Result};

const PAGE: usize = 4096;

/// One chip's memory window and the share of it the device's PEs own, an eighth per PE: the
/// firmware image at the share's top, everything below it for the allocator.
pub(crate) struct Dram {
    pub(crate) chip: u8,
    share: Range<usize>,
    /// Where the firmware image is staged in the window.
    pub(crate) offset: usize,
    /// Where a cluster sees this window, once a boot found the image in it.
    pub(crate) at: Option<u64>,
}

impl Dram {
    /// The share of `chip`'s memory the PE bitmask `pes` owns, with `image` placed at its top;
    /// The returned placement leaves space for firmware DMA descriptors.
    pub(crate) fn open(chip: u8, pes: u8, image: usize) -> Result<Self> {
        let bar = open_rw(&node(chip, "bar4"))?;
        let mut usable = rngd::BarInfo::default();
        // SAFETY: `usable` is a valid writable `BarInfo` for this ioctl.
        unsafe { rngd::available_dram(bar.as_raw_fd(), &mut usable) }
            .map_err(|why| Error::Device(format!("asking for the usable window: {why}")))?;
        let chip_pes = usize::from(rngd::PES);
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
            share,
            offset,
            at: None,
        })
    }

    /// Where a cluster may see this window: the located address once one cluster booted, every
    /// candidate before.
    pub(crate) fn candidates(&self) -> Vec<u64> {
        self.at.map_or(rngd::CANDIDATES.to_vec(), |at| vec![at])
    }

    /// The offsets of this window the allocator may hand out: the share below the staged image.
    pub(crate) fn free(&self) -> Range<usize> {
        self.share.start..self.offset - rngd::PDMA_DESCRIPTORS
    }
}
