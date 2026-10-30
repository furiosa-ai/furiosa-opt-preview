//! The compiler image: device code, memory bindings and source metadata, in the format the
//! compiler writes into a registry [`Entry`](crate::registry::Entry) and the runtime loads.
//!
//! - The driver encodes images with its own build of this module, and the SDK of the same release parses
//!   them; the magic names the format, so an image of another format is refused, not misread.
//! - [`Level`] and [`MARKER_IDS`] are the profile contract with the compiler's device code.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ops::Deref;

use bincode::{BorrowDecode, Decode, Encode};

use crate::arch::rngd::{self, sram};
use crate::executable::Executable;
use crate::{Topology, memory};

/// A compiler image whose contents passed every rule a load relies on. It is read through
/// [`Parts`] and made only from parts that pass, so no other image exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image<'a>(Parts<'a>);

impl<'a> Image<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        let body = bytes.strip_prefix(MAGIC).ok_or("not a compiler image of this format")?;
        let (parts, read): (Parts<'a>, _) =
            bincode::borrow_decode_from_slice(body, CONFIG).map_err(|error| error.to_string())?;
        if read != body.len() {
            return Err("trailing bytes after the compiler image".into());
        }
        parts.try_into()
    }

    pub fn encode(&self) -> Result<Vec<u8>, String> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(bincode::encode_to_vec(&self.0, CONFIG).map_err(|error| error.to_string())?);
        Ok(bytes)
    }

    pub fn into_parts(self) -> Parts<'a> {
        self.0
    }
}

impl<'a> TryFrom<Parts<'a>> for Image<'a> {
    type Error = String;

    fn try_from(parts: Parts<'a>) -> Result<Self, String> {
        let Topology { chips, pes } = parts.topology;
        if !(1..=rngd::MAX_CHIPS).contains(&chips) || !(1..=rngd::PES).contains(&pes) {
            return Err(format!("topology {chips}x{pes} is not an RNGD device"));
        }
        let columns = parts.topology.clusters();
        if parts.executable.clusters() != columns {
            return Err("executable columns do not match the topology".into());
        }
        if parts
            .weight
            .is_some_and(|weight| weight.bytes.is_empty() || !weight.bytes.len().is_multiple_of(usize::from(chips)))
        {
            return Err("initial data is empty or not divisible by chip count".into());
        }
        if parts.sram_end > sram::WINDOW.end {
            return Err("temporaries exceed SRAM capacity".into());
        }
        let named = |slot| parts.inputs.contains(&slot) || parts.outputs.contains(&slot);
        if parts
            .inputs
            .iter()
            .chain(&parts.outputs)
            .any(|&slot| slot as usize >= parts.slots.len())
            || !(0..parts.slots.len() as u32).all(named)
        {
            return Err("argument slots do not match the signature".into());
        }
        if parts.profile.spans.iter().any(|span| {
            span.chunk as usize >= parts.executable.chunks()
                || span.cluster as usize >= columns
                || span.begin >= MARKER_IDS
                || span.end >= MARKER_IDS
        }) {
            return Err("profile span names an invalid marker".into());
        }
        Ok(Self(parts))
    }
}

impl<'a> Deref for Image<'a> {
    type Target = Parts<'a>;

    fn deref(&self) -> &Parts<'a> {
        &self.0
    }
}

/// An image's contents before their rules are checked; an [`Image`] holds only contents that pass.
///
/// Device code reads its operands from a table in a fixed order, the compiler's `enum dram_kind`:
/// the weight, a reserved operand, the two stacks (local, then inter-chip; an empty stack has no
/// buffer), then `slots` in signature order. A launch binds `inputs` first, then the `outputs` not
/// already among them; outputs are memory the caller provides.
#[derive(Encode, BorrowDecode, Clone, Debug, PartialEq, Eq)]
pub struct Parts<'a> {
    pub topology: Topology,
    pub executable: Executable<'a>,
    pub weight: Option<Weight<'a>>,
    pub stacks: [Stack; 2],
    pub slots: Vec<Slot>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub profile: Profile,
    pub sram_end: u64,
}

#[derive(Encode, BorrowDecode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weight<'a> {
    pub kind: DramKind,
    /// Every chip's shard in chip order; never empty.
    pub bytes: &'a [u8],
}

/// The DRAM a compiler static lives in: statics never live in SRAM.
#[derive(Encode, Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DramKind {
    Dram,
    DramShared,
}

impl From<DramKind> for memory::Kind {
    fn from(kind: DramKind) -> Self {
        match kind {
            DramKind::Dram => Self::Dram,
            DramKind::DramShared => Self::DramShared,
        }
    }
}

/// One of the compiler's stacks.
#[derive(Encode, Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stack {
    pub kind: DramKind,
    pub size: u64,
}

#[derive(Encode, Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub kind: memory::Kind,
    pub size: u64,
}

#[derive(Encode, Decode, Clone, Debug, Default, PartialEq, Eq)]
pub struct Profile {
    /// The most hits one chunk records at each of [`Level::ALL`].
    pub depth: [u32; 3],
    pub spans: Vec<Span>,
}

impl Profile {
    /// The most hits one chunk records at `level`.
    pub fn depth(&self, level: Level) -> u32 {
        self.depth[level as usize - Level::Info as usize]
    }
}

/// Profile levels retain the compiler's `ProfileLevel` and libpe's `tuc_profile_level` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    pub const ALL: [Self; 3] = [Self::Info, Self::Debug, Self::Trace];
}

/// A span the device brackets with two markers, in one chunk of one cluster column.
#[derive(Encode, Decode, Clone, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub chunk: u32,
    /// The cluster column across all chips.
    pub cluster: u32,
    /// The marker ids that open and close the span.
    pub begin: u16,
    pub end: u16,
    /// Whether `end` bounds the span's end from above instead of marking it, as for work on a
    /// tensor-unit context.
    pub upper_bound: bool,
    pub name: String,
    /// The LIR instruction the span profiles, as `DeepInstructionIndex` displays it.
    pub instruction: Option<String>,
}

/// Marker ids a profile hit can carry, its low 14 bits; the compiler's `MAX_PROFILE_UID` keeps
/// the ids it assigns below this.
pub const MARKER_IDS: u16 = 1 << 14;

/// `FOPT` named the format earlier releases wrote.
const MAGIC: &[u8; 4] = b"FOP2";
const CONFIG: bincode::config::Configuration = bincode::config::standard();

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn parts() -> Parts<'static> {
        let executable = Executable::encode(&[vec![b"code"; 4]]);
        Parts {
            topology: Topology { chips: 2, pes: 8 },
            executable: Executable::parse(executable.unwrap().leak()).unwrap(),
            weight: Some(Weight {
                kind: DramKind::Dram,
                bytes: b"weights!",
            }),
            stacks: [Stack {
                kind: DramKind::Dram,
                size: 0,
            }; 2],
            slots: vec![Slot {
                kind: memory::Kind::DramShared,
                size: 64,
            }],
            inputs: vec![],
            outputs: vec![0],
            profile: Profile {
                depth: [2; 3],
                spans: vec![Span {
                    cluster: 3,
                    name: "task".into(),
                    begin: 1,
                    end: 2,
                    instruction: Some("7".into()),
                    upper_bound: true,
                    ..Default::default()
                }],
            },
            sram_end: 1024,
        }
    }

    /// An image parses back to itself; a truncated, extended, foreign or earlier-format one is
    /// refused.
    #[test]
    fn round_trips_only_complete_images() {
        let image = Image::try_from(parts()).unwrap();
        let mut bytes = image.encode().unwrap();
        let decoded = Image::parse(&bytes).unwrap();
        assert_eq!(decoded, image);
        for len in 0..bytes.len() {
            assert!(Image::parse(&bytes[..len]).is_err());
        }
        bytes.push(0);
        assert!(Image::parse(&bytes).is_err());
        bytes.pop();
        bytes[..4].copy_from_slice(b"FOPT");
        assert!(Image::parse(&bytes).is_err());
        bytes[0] = b'X';
        assert!(Image::parse(&bytes).is_err());
    }

    /// Executable serialization preserves byte-slice encoding; decoding refuses malformed code.
    #[test]
    fn executable_codec_preserves_bytes() {
        let bytes = Executable::encode(&[vec![b"code"; 4]]).unwrap();
        let executable = Executable::parse(&bytes).unwrap();
        let encoded = bincode::encode_to_vec(executable, CONFIG).unwrap();
        assert_eq!(encoded, bincode::encode_to_vec(bytes.as_slice(), CONFIG).unwrap());
        let (decoded, _): (Executable<'_>, _) = bincode::borrow_decode_from_slice(&encoded, CONFIG).unwrap();
        assert_eq!(decoded.bytes(), bytes.as_slice());
        let mut invalid = bytes;
        invalid[0] = b'X';
        let encoded = bincode::encode_to_vec(invalid.as_slice(), CONFIG).unwrap();
        assert!(bincode::borrow_decode_from_slice::<Executable<'_>, _>(&encoded, CONFIG).is_err());
    }

    /// No image holds slots outside the signature, spans naming no cluster or marker, or empty
    /// initial data.
    #[test]
    fn rejects_invalid_bindings_and_markers() {
        let refused = |edit: fn(&mut Parts<'static>)| {
            let mut parts = parts();
            edit(&mut parts);
            Image::try_from(parts).err()
        };
        let reason = |reason: &str| Some(reason.to_string());
        let slots = reason("argument slots do not match the signature");
        let marker = reason("profile span names an invalid marker");
        assert_eq!(refused(|parts| parts.outputs = vec![1]), slots);
        assert_eq!(refused(|parts| parts.outputs = vec![]), slots);
        assert_eq!(refused(|parts| parts.profile.spans[0].cluster = 4), marker);
        assert_eq!(refused(|parts| parts.profile.spans[0].end = MARKER_IDS), marker);
        assert_eq!(
            refused(|parts| {
                parts.weight = Some(Weight {
                    kind: DramKind::Dram,
                    bytes: &[],
                })
            }),
            reason("initial data is empty or not divisible by chip count")
        );
    }
}
