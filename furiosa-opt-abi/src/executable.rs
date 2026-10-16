//! Device entry ranges, independent of compiler arguments and metadata.

use core::ops::Range;

const MAGIC: &[u8; 4] = b"FOE1";
const HEADER: usize = 16;
const ENTRY: usize = 16;
const ALIGNMENT: usize = 256;

/// Preferred chunk budget; an indivisible instruction may use the full reserved code region.
pub const CHUNK_LIMIT: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid device executable")]
pub struct Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Executable<'a> {
    bytes: &'a [u8],
    table: usize,
    stride: usize,
    chunks: usize,
    clusters: usize,
}

/// One cluster's code for one chunk, and where it is entered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub code: Range<usize>,
    pub entry: usize,
}

impl Chunk {
    /// The code's length in whole alignment units, as the encoder pads it.
    pub fn padded_len(&self) -> usize {
        self.code.len().next_multiple_of(ALIGNMENT)
    }
}

impl<'a> Executable<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.get(..4) != Some(MAGIC) || bytes.len() < HEADER {
            return Err(Error);
        }
        let executable = Self {
            bytes,
            table: u16::from_le_bytes(bytes[4..6].try_into().unwrap()) as usize,
            stride: u16::from_le_bytes(bytes[6..8].try_into().unwrap()) as usize,
            chunks: u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            clusters: u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize,
        };
        if executable.table < HEADER || executable.stride < ENTRY || executable.chunks == 0 || executable.clusters == 0
        {
            return Err(Error);
        }
        let count = executable.chunks.checked_mul(executable.clusters).ok_or(Error)?;
        let mut end = executable
            .table
            .checked_add(count.checked_mul(executable.stride).ok_or(Error)?)
            .filter(|end| *end <= bytes.len())
            .ok_or(Error)?;
        for index in 0..count {
            let chunk = executable.at(index);
            if chunk.code.start < end
                || !chunk.code.start.is_multiple_of(ALIGNMENT)
                || chunk.code.end > bytes.len()
                || chunk.entry >= chunk.code.len()
                || !chunk.entry.is_multiple_of(4)
            {
                return Err(Error);
            }
            end = chunk.code.end;
        }
        Ok(executable)
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn chunks(&self) -> usize {
        self.chunks
    }

    pub fn clusters(&self) -> usize {
        self.clusters
    }

    /// One cluster's chunks, in run order.
    pub fn code(self, cluster: usize) -> Result<impl Iterator<Item = Chunk> + 'a, Error> {
        if cluster >= self.clusters {
            return Err(Error);
        }
        Ok((0..self.chunks).map(move |chunk| self.at(chunk * self.clusters + cluster)))
    }

    /// Table entry `index`; an offset past `usize` saturates, so [`Self::parse`] rejects it.
    fn at(&self, index: usize) -> Chunk {
        let words = &self.bytes[self.table + index * self.stride..][..ENTRY];
        let offset = usize::try_from(u64::from_le_bytes(words[..8].try_into().unwrap())).unwrap_or(usize::MAX);
        let len = u32::from_le_bytes(words[8..12].try_into().unwrap()) as usize;
        Chunk {
            code: offset..offset.saturating_add(len),
            entry: u32::from_le_bytes(words[12..16].try_into().unwrap()) as usize,
        }
    }

    /// Encodes `chunks[chunk][cluster]`, every chunk with the same cluster count.
    #[cfg(feature = "alloc")]
    pub fn encode(chunks: &[alloc::vec::Vec<&[u8]>]) -> Result<alloc::vec::Vec<u8>, Error> {
        let clusters = chunks.first().ok_or(Error)?.len();
        if clusters == 0 || chunks.iter().any(|chunk| chunk.len() != clusters) {
            return Err(Error);
        }
        let count = chunks.len().checked_mul(clusters).ok_or(Error)?;
        let table = HEADER
            .checked_add(count.checked_mul(ENTRY).ok_or(Error)?)
            .ok_or(Error)?;
        let mut bytes = alloc::vec![0; table];
        bytes[..4].copy_from_slice(MAGIC);
        bytes[4..6].copy_from_slice(&(HEADER as u16).to_le_bytes());
        bytes[6..8].copy_from_slice(&(ENTRY as u16).to_le_bytes());
        bytes[8..12].copy_from_slice(&u32::try_from(chunks.len()).map_err(|_| Error)?.to_le_bytes());
        bytes[12..16].copy_from_slice(&u32::try_from(clusters).map_err(|_| Error)?.to_le_bytes());
        for (index, code) in chunks.iter().flatten().enumerate() {
            if code.is_empty() {
                return Err(Error);
            }
            bytes.resize(bytes.len().next_multiple_of(ALIGNMENT), 0);
            let offset = bytes.len() as u64;
            let len = u32::try_from(code.len()).map_err(|_| Error)?;
            let at = HEADER + index * ENTRY;
            bytes[at..at + 8].copy_from_slice(&offset.to_le_bytes());
            bytes[at + 8..at + 12].copy_from_slice(&len.to_le_bytes());
            bytes.extend_from_slice(code);
        }
        bytes.resize(bytes.len().next_multiple_of(ALIGNMENT), 0);
        Ok(bytes)
    }
}

#[cfg(feature = "serde")]
mod codec {
    use super::Executable;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

    impl Serialize for Executable<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.bytes.serialize(serializer)
        }
    }

    impl<'de: 'a, 'a> Deserialize<'de> for Executable<'a> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let bytes = <&[u8]>::deserialize(deserializer)?;
            Self::parse(bytes).map_err(de::Error::custom)
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec;

    /// FOE1 gives each cluster its chunks and entries in run order, and refuses malformed tables.
    #[test]
    fn preserves_each_clusters_chunks() {
        let chunks = vec![vec![&b"abcd"[..], &b"efghijkl"[..]], vec![&b"mnop"[..], &b"qrst"[..]]];
        let mut bytes = Executable::encode(&chunks).unwrap();
        bytes[HEADER + ENTRY + 12..][..4].copy_from_slice(&4_u32.to_le_bytes());
        let executable = Executable::parse(&bytes).unwrap();
        for cluster in 0..2 {
            let code = executable
                .code(cluster)
                .unwrap()
                .map(|chunk| (&bytes[chunk.code], chunk.entry));
            let entries = chunks.iter().zip([[0, 4], [0, 0]]);
            assert!(code.eq(entries.map(|(code, entry)| (code[cluster], entry[cluster]))));
        }
        assert!(executable.code(2).is_err());
        for offset in [0u64, 256, u64::MAX] {
            let mut invalid = bytes.clone();
            invalid[HEADER + ENTRY..HEADER + ENTRY + 8].copy_from_slice(&offset.to_le_bytes());
            assert!(Executable::parse(&invalid).is_err());
        }
        for len in 0..bytes.len() - ALIGNMENT {
            assert!(Executable::parse(&bytes[..len]).is_err());
        }
    }
}
