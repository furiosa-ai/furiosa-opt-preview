//! Host writes to local HBM through BAR4.

use std::fs::File;
use std::sync::Mutex;

use super::map::Mapping;
use crate::Result;
use crate::arch::rngd::BAR4_DRAM;

const PAGE: usize = 4096;

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
