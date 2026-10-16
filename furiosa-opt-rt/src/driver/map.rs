//! Shared mappings of device memory and driver regions.

use std::fs::{File, OpenOptions};
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use nix::sys::mman::{MapFlags, ProtFlags, mmap, munmap};

use crate::{Error, Result};

pub(crate) fn open_rw(path: &str) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|why| Error::Device(format!("{path}: {why}")))
}

pub(crate) struct Mapping {
    pub(crate) at: NonNull<u8>,
    pub(crate) len: usize,
}

// SAFETY: owners serialize mapping access through an exclusive borrow or a mutex.
unsafe impl Send for Mapping {}

impl Mapping {
    pub(crate) fn new(file: &File, offset: i64, len: usize) -> Result<Self> {
        let len =
            NonZeroUsize::new(len).ok_or_else(|| Error::Device("a mapping of no bytes addresses nothing".into()))?;
        // SAFETY: the driver maps device memory here and the length is checked against the region's own size.
        let at = unsafe {
            mmap(
                None,
                len,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                file,
                offset,
            )
        }
        .map_err(|why| Error::Device(format!("mapping at {offset:#x} failed: {why}")))?;
        Ok(Self {
            at: at.cast(),
            len: len.get(),
        })
    }
}

impl Mapping {
    /// Writes `bytes` at `offset`: whole words from the first aligned address, since word stores
    /// post whole over the uncached mapping, and the ragged ends bytewise. Panics past the mapping.
    pub(crate) fn store(&self, offset: usize, bytes: &[u8]) {
        assert!(
            offset.checked_add(bytes.len()).is_some_and(|end| end <= self.len),
            "{} bytes at {offset:#x} fall outside a {:#x} byte mapping",
            bytes.len(),
            self.len
        );
        let target = self.at.as_ptr().wrapping_add(offset);
        let head = head(target.addr(), bytes.len());
        let (words, tail) = bytes[head..].as_chunks::<WORD>();
        // SAFETY: the range is inside the mapping, checked above, and `head` moves the word stores
        // to an aligned address.
        unsafe {
            for (index, byte) in bytes[..head].iter().enumerate() {
                target.add(index).write_volatile(*byte);
            }
            for (index, word) in words.iter().enumerate() {
                target
                    .add(head)
                    .cast::<u64>()
                    .add(index)
                    .write_volatile(u64::from_ne_bytes(*word));
            }
            for (index, byte) in tail.iter().enumerate() {
                target.add(head + words.len() * WORD + index).write_volatile(*byte);
            }
        }
    }
}

const WORD: usize = size_of::<u64>();

/// Bytes at `addr` before the first word-aligned address, at most `len`.
const fn head(addr: usize, len: usize) -> usize {
    let head = addr.wrapping_neg() % WORD;
    if head < len { head } else { len }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this is the mapping made in `new` and nothing else unmaps it.
        let _ = unsafe { munmap(self.at.cast(), self.len) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A write of any length and offset lands byte for byte in BAR memory.
    #[test]
    fn unaligned_store_lands_whole() {
        let file = tempfile::tempfile().expect("a file to map");
        file.set_len(4096).expect("a page to map");
        let mapping = Mapping::new(&file, 0, 4096).expect("mapping");
        let bytes = (1..=19).collect::<Vec<u8>>();

        mapping.store(1, &bytes);

        let mut landed = vec![0; 32];
        std::os::unix::fs::FileExt::read_exact_at(&file, &mut landed, 0).expect("read back");
        assert_eq!(&landed[1..20], bytes.as_slice());
        assert_eq!((landed[0], landed[20]), (0, 0));
    }
}
