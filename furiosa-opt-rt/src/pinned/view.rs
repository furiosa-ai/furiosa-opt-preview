use super::Backing;

/// Immutable host bytes with their DMA allocation identity, when pinned.
#[derive(Clone, Debug)]
pub struct HostBufferSlice<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) backing: Option<Backing<'a>>,
}

impl<'a> HostBufferSlice<'a> {
    /// Consecutive pieces of `len` bytes (one byte when `len` is zero), each keeping its place in
    /// the DMA-buf.
    pub fn chunks(self, len: usize) -> impl Iterator<Item = Self> {
        let Self { bytes, backing } = self;
        bytes.chunks(len.max(1)).map(move |piece| Self {
            bytes: piece,
            backing: backing.map(|backing| backing.piece(bytes.as_ptr().addr(), piece.as_ptr().addr())),
        })
    }

    /// The bytes the view covers.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Bytes the view covers.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the view covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl<'a> From<&'a [u8]> for HostBufferSlice<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self { bytes, backing: None }
    }
}

/// Exclusively borrowed host bytes with their DMA allocation identity, when pinned.
#[derive(Debug)]
pub struct HostBufferSliceMut<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) backing: Option<Backing<'a>>,
}

impl<'a> HostBufferSliceMut<'a> {
    /// Consecutive pieces of `len` bytes (one byte when `len` is zero), each keeping its place in
    /// the DMA-buf.
    pub fn chunks(self, len: usize) -> impl Iterator<Item = Self> {
        let Self { bytes, backing } = self;
        let start = bytes.as_ptr().addr();
        bytes.chunks_mut(len.max(1)).map(move |piece| Self {
            backing: backing.map(|backing| backing.piece(start, piece.as_ptr().addr())),
            bytes: piece,
        })
    }

    /// The bytes the view covers.
    pub fn into_bytes(self) -> &'a mut [u8] {
        self.bytes
    }

    /// Bytes the view covers.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the view covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl<'a> From<&'a mut [u8]> for HostBufferSliceMut<'a> {
    fn from(bytes: &'a mut [u8]) -> Self {
        Self { bytes, backing: None }
    }
}
