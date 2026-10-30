//! Host stores into a chip's DRAM through its BAR4.

use std::fs::File;
use std::sync::Mutex;

use super::BAR4_DRAM;
use super::map::Mapping;
use crate::Result;

/// A chip's DRAM as the host's CPU reaches it: the kernel driver exposes the chip's DRAM in BAR4
/// from [`BAR4_DRAM`] on, and a store there lands in DRAM with no DMA command. Small writes take it,
/// since a store costs less than a DMA round trip. It maps one page-aligned span at a time, the one
/// the last write needed, and maps anew when a write falls outside.
pub(crate) struct Window {
    file: File,
    mapping: Mutex<Option<(u64, Mapping)>>,
}

impl Window {
    pub(crate) fn new(file: File) -> Self {
        Self {
            file,
            mapping: Mutex::new(None),
        }
    }

    pub(crate) fn write(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        let offset = BAR4_DRAM + offset;
        let page = offset & !(PAGE as u64 - 1);
        let end = (offset - page) as usize + bytes.len();
        let mut mapping = self.mapping.lock().unwrap();
        if mapping
            .as_ref()
            .is_none_or(|(base, mapping)| *base != page || mapping.len < end)
        {
            *mapping = Some((page, Mapping::new(&self.file, page as i64, end.next_multiple_of(PAGE))?));
        }
        let mapping = &mapping.as_ref().unwrap().1;
        let offset = (offset - page) as usize;
        mapping.store(offset, bytes);
        if !bytes.is_empty() {
            // A read from the same endpoint drains posted writes before access is released.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            // SAFETY: the last written byte is inside the mapping and this caller owns write access.
            unsafe { mapping.at.as_ptr().add(offset + bytes.len() - 1).read_volatile() };
        }
        Ok(())
    }
}

const PAGE: usize = 4096;
