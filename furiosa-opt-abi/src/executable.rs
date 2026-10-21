//! Device entry ranges, independent of compiler arguments and metadata.

use core::ops::Range;

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// A device executable [`Executable::parse`] checked: each chunk's code lies aligned and in
/// order inside its bytes, and its entry inside that code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Executable<'a> {
    bytes: &'a [u8],
    table: usize,
    stride: usize,
    chunks: usize,
    clusters: usize,
}

impl<'a> Executable<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let (header, _) = Header::ref_from_prefix(bytes).map_err(|_| Error)?;
        if header.magic != MAGIC {
            return Err(Error);
        }
        let executable = Self {
            bytes,
            table: usize::from(header.table.get()),
            stride: usize::from(header.stride.get()),
            chunks: header.chunks.get() as usize,
            clusters: header.clusters.get() as usize,
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

    /// Encodes `chunks[chunk][cluster]`, every chunk with the same cluster count.
    #[cfg(feature = "alloc")]
    pub fn encode(chunks: &[alloc::vec::Vec<&[u8]>]) -> Result<alloc::vec::Vec<u8>, Error> {
        let clusters = chunks.first().ok_or(Error)?.len();
        if clusters == 0 || chunks.iter().any(|chunk| chunk.len() != clusters) {
            return Err(Error);
        }
        let count = chunks.len().checked_mul(clusters).ok_or(Error)?;
        let mut end = HEADER
            .checked_add(count.checked_mul(ENTRY).ok_or(Error)?)
            .ok_or(Error)?;
        let mut entries = alloc::vec::Vec::with_capacity(count);
        for code in chunks.iter().flatten() {
            if code.is_empty() {
                return Err(Error);
            }
            let offset = end.next_multiple_of(ALIGNMENT);
            entries.push(Entry {
                offset: U64::new(offset as u64),
                len: U32::new(u32::try_from(code.len()).map_err(|_| Error)?),
                entry: U32::ZERO,
            });
            end = offset + code.len();
        }
        let header = Header {
            magic: MAGIC,
            table: U16::new(HEADER as u16),
            stride: U16::new(ENTRY as u16),
            chunks: U32::new(u32::try_from(chunks.len()).map_err(|_| Error)?),
            clusters: U32::new(u32::try_from(clusters).map_err(|_| Error)?),
        };
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(entries.as_bytes());
        for (entry, code) in entries.iter().zip(chunks.iter().flatten()) {
            bytes.resize(entry.offset.get() as usize, 0);
            bytes.extend_from_slice(code);
        }
        bytes.resize(bytes.len().next_multiple_of(ALIGNMENT), 0);
        Ok(bytes)
    }

    /// One cluster's chunks, in run order.
    pub fn code(self, cluster: usize) -> Result<impl Iterator<Item = Chunk> + 'a, Error> {
        if cluster >= self.clusters {
            return Err(Error);
        }
        Ok((0..self.chunks).map(move |chunk| self.at(chunk * self.clusters + cluster)))
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

    /// Table entry `index`; an offset past `usize` saturates, so [`Self::parse`] rejects it.
    fn at(&self, index: usize) -> Chunk {
        let (entry, _) = Entry::ref_from_prefix(&self.bytes[self.table + index * self.stride..])
            .unwrap_or_else(|_| unreachable!("parse checked that the table lies inside the bytes"));
        let offset = usize::try_from(entry.offset.get()).unwrap_or(usize::MAX);
        Chunk {
            code: offset..offset.saturating_add(entry.len.get() as usize),
            entry: entry.entry.get() as usize,
        }
    }
}

/// One cluster's code for one chunk, and where it is entered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub code: Range<usize>,
    pub entry: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid device executable")]
pub struct Error;

/// Every chunk's code starts on this boundary, and the bytes up to the next one are padding.
pub const ALIGNMENT: usize = 256;

/// Preferred chunk budget; an indivisible instruction may use the full reserved code region.
pub const CHUNK_LIMIT: usize = 1 << 20;

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

/// The executable's first bytes.
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
struct Header {
    magic: [u8; 4],
    /// Where the entry table starts.
    table: U16,
    /// Bytes from one table entry to the next.
    stride: U16,
    chunks: U32,
    clusters: U32,
}

/// One table entry: where one cluster's code for one chunk lies, and where it is entered.
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
struct Entry {
    offset: U64,
    len: U32,
    entry: U32,
}

const MAGIC: [u8; 4] = *b"FOE1";
const HEADER: usize = size_of::<Header>();
const ENTRY: usize = size_of::<Entry>();

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec;

    /// FOE1 gives each cluster its chunks and entries in run order, and refuses malformed tables.
    #[test]
    fn preserves_each_clusters_chunks() {
        let chunks = vec![vec![&b"abcd"[..], &b"efghijkl"[..]], vec![&b"mnop"[..], &b"qrst"[..]]];
        let mut bytes = Executable::encode(&chunks).unwrap();
        assert_eq!(
            bytes[..HEADER],
            *b"FOE1\x10\x00\x10\x00\x02\x00\x00\x00\x02\x00\x00\x00"
        );
        fn entry(bytes: &mut [u8], index: usize) -> &mut Entry {
            Entry::mut_from_prefix(&mut bytes[HEADER + index * ENTRY..]).unwrap().0
        }
        entry(&mut bytes, 1).entry = U32::new(4);
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
            entry(&mut invalid, 1).offset = U64::new(offset);
            assert!(Executable::parse(&invalid).is_err());
        }
        for len in 0..bytes.len() - ALIGNMENT {
            assert!(Executable::parse(&bytes[..len]).is_err());
        }
    }
}
