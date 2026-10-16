use super::Backing;

/// Immutable host bytes with their DMA allocation identity, when pinned.
#[derive(Clone, Debug)]
pub struct HostBufferSlice<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) backing: Option<Backing<'a>>,
}

/// Exclusively borrowed host bytes with their DMA allocation identity, when pinned.
#[derive(Debug)]
pub struct HostBufferSliceMut<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) backing: Option<Backing<'a>>,
}

impl<'a> HostBufferSlice<'a> {
    /// Bytes the view covers.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the view covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The bytes the view covers.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Consecutive pieces of `len` bytes (one byte when `len` is zero), each keeping its place in
    /// the DMA-buf.
    pub fn chunks(self, len: usize) -> impl Iterator<Item = Self> {
        let Self { bytes, backing } = self;
        bytes.chunks(len.max(1)).map(move |piece| Self {
            bytes: piece,
            backing: backing.map(|backing| backing.piece(bytes.as_ptr().addr(), piece.as_ptr().addr())),
        })
    }
}

impl<'a> HostBufferSliceMut<'a> {
    /// Bytes the view covers.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the view covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The bytes the view covers.
    pub fn into_bytes(self) -> &'a mut [u8] {
        self.bytes
    }

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
}

impl<'a> From<&'a [u8]> for HostBufferSlice<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self { bytes, backing: None }
    }
}

impl<'a> From<&'a mut [u8]> for HostBufferSliceMut<'a> {
    fn from(bytes: &'a mut [u8]) -> Self {
        Self { bytes, backing: None }
    }
}

#[cfg(test)]
mod tests {
    use crate::test::pinned;

    /// Each chunk of a pinned range names its own bytes and where they sit in the DMA-buf.
    #[test]
    fn chunks_keep_their_offsets() {
        let mut host = pinned();
        let read: Vec<_> = host
            .slice(1024..3072)
            .chunks(1024)
            .map(|c| (c.bytes.len(), c.backing.map(|b| b.offset)))
            .collect();
        assert_eq!(read, [(1024, Some(1024)), (1024, Some(2048))]);
        let write: Vec<_> = host
            .slice_mut(..2048)
            .chunks(1024)
            .map(|c| c.backing.map(|b| b.offset))
            .collect();
        assert_eq!(write, [Some(0), Some(1024)]);
        let bytes: Vec<_> = host
            .slice(1..4)
            .chunks(0)
            .map(|c| c.backing.map(|b| b.offset))
            .collect();
        assert_eq!(bytes, [Some(1), Some(2), Some(3)]);
    }
}
