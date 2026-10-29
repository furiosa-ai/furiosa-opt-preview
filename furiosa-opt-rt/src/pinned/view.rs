use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use super::Backing;
use crate::buffer::Region;

/// Immutable host bytes with their DMA allocation identity, when pinned.
#[derive(Clone)]
pub struct HostBufferSlice<'a> {
    pub(crate) address: usize,
    pub(crate) len: usize,
    pub(crate) pinned: Option<Backing<'a>>,
    pub(crate) borrow: PhantomData<&'a [u8]>,
}

impl<'a> HostBufferSlice<'a> {
    /// Consecutive pieces of `len` bytes (one byte when `len` is zero), each keeping its place in
    /// the DMA-buf.
    pub fn chunks(self, len: usize) -> impl Iterator<Item = Self> {
        let step = len.max(1);
        (0..self.len).step_by(step).map(move |at| Self {
            address: self.address + at,
            len: step.min(self.len - at),
            pinned: self
                .pinned
                .map(|backing| backing.piece(self.address, self.address + at)),
            borrow: PhantomData,
        })
    }

    /// The bytes the view covers.
    pub fn bytes(&self) -> Bytes<'_> {
        if let Some(backing) = self.pinned {
            Region::host(backing, self.len, crate::Access::Read).wait();
        }
        // SAFETY: earlier device writes ended, and this shared borrow keeps later ones off the bytes.
        let bytes = unsafe { std::slice::from_raw_parts(std::ptr::with_exposed_provenance(self.address), self.len) };
        Bytes { bytes }
    }

    /// Bytes the view covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a> From<&'a [u8]> for HostBufferSlice<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self {
            address: bytes.as_ptr().expose_provenance(),
            len: bytes.len(),
            pinned: None,
            borrow: PhantomData,
        }
    }
}

/// Exclusively borrowed host bytes with their DMA allocation identity, when pinned.
pub struct HostBufferSliceMut<'a> {
    pub(crate) address: usize,
    pub(crate) len: usize,
    pub(crate) pinned: Option<Backing<'a>>,
    pub(crate) borrow: PhantomData<&'a mut [u8]>,
}

impl<'a> HostBufferSliceMut<'a> {
    /// Consecutive pieces of `len` bytes (one byte when `len` is zero), each keeping its place in
    /// the DMA-buf.
    pub fn chunks(self, len: usize) -> impl Iterator<Item = Self> {
        let step = len.max(1);
        (0..self.len).step_by(step).map(move |at| Self {
            address: self.address + at,
            len: step.min(self.len - at),
            pinned: self
                .pinned
                .map(|backing| backing.piece(self.address, self.address + at)),
            borrow: PhantomData,
        })
    }

    /// The bytes before `offset` and from it on, each keeping its place in the DMA-buf; an
    /// `offset` past the end panics, as [`slice::split_at_mut`] does.
    pub fn split_at(self, offset: usize) -> (Self, Self) {
        assert!(offset <= self.len, "split at {offset} of {} host bytes", self.len);
        let piece = |at: usize, len: usize| Self {
            address: self.address + at,
            len,
            pinned: self
                .pinned
                .map(|backing| backing.piece(self.address, self.address + at)),
            borrow: PhantomData,
        };
        (piece(0, offset), piece(offset, self.len - offset))
    }

    /// The bytes the view covers.
    pub fn into_bytes(self) -> BytesMut<'a> {
        if let Some(backing) = self.pinned {
            Region::host(backing, self.len, crate::Access::Write).wait();
        }
        // SAFETY: earlier device accesses ended, and this exclusive borrow keeps later ones off the bytes.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(std::ptr::with_exposed_provenance_mut(self.address), self.len) };
        BytesMut { bytes }
    }

    /// Bytes the view covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a> From<&'a mut [u8]> for HostBufferSliceMut<'a> {
    fn from(bytes: &'a mut [u8]) -> Self {
        Self {
            address: bytes.as_mut_ptr().expose_provenance(),
            len: bytes.len(),
            pinned: None,
            borrow: PhantomData,
        }
    }
}

/// CPU-readable bytes; their borrow keeps device writes off them.
pub struct Bytes<'a> {
    bytes: &'a [u8],
}

impl Deref for Bytes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.bytes
    }
}

/// CPU-writable bytes; their borrow keeps device accesses off them.
pub struct BytesMut<'a> {
    bytes: &'a mut [u8],
}

impl Deref for BytesMut<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.bytes
    }
}

impl DerefMut for BytesMut<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.bytes
    }
}

/// The view's address and length: reading its bytes could wait for device work or find them poisoned.
impl std::fmt::Debug for HostBufferSlice<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (formatter.debug_struct("HostBufferSlice"))
            .field("address", &format_args!("{:#x}", self.address))
            .field("len", &self.len)
            .finish()
    }
}

/// The view's address and length: reading its bytes could wait for device work or find them poisoned.
impl std::fmt::Debug for HostBufferSliceMut<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (formatter.debug_struct("HostBufferSliceMut"))
            .field("address", &format_args!("{:#x}", self.address))
            .field("len", &self.len)
            .finish()
    }
}
