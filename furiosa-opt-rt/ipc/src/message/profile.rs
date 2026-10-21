//! A launch's TUC profile: what the request asks every chunk to record, and the hits the reply
//! carries back.

use alloc::vec::Vec;
use bincode::de::{BorrowDecoder, Decoder};
use bincode::error::DecodeError;
use bincode::{BorrowDecode, Decode, Encode};

/// Profiles a launch at the task library's `level`, `Info` (3) through `Trace` (5), with the TUC
/// keeping up to `depth` hits of each chunk, at most [`Profile::QUEUE`]. Exists only in that range.
#[derive(Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    level: u8,
    depth: u32,
}

impl Profile {
    /// Hits the firmware's TUC profile queue holds.
    pub const QUEUE: u32 = 8192;
    /// Hits one reply carries, every chunk's together.
    pub const HITS: u32 = 8192;
    /// Chunks one reply carries the hits of.
    pub const CHUNKS: u32 = 256;

    pub fn new(level: u8, depth: u32) -> Option<Self> {
        ((3..=5).contains(&level) && (1..=Self::QUEUE).contains(&depth)).then_some(Self { level, depth })
    }

    /// Whether one reply carries every hit a launch of `chunks` chunks keeps.
    pub fn fits(self, chunks: usize) -> bool {
        chunks <= Self::CHUNKS as usize && chunks * self.depth as usize <= Self::HITS as usize
    }

    pub fn level(self) -> u8 {
        self.level
    }

    pub fn depth(self) -> u32 {
        self.depth
    }
}

impl<Context> Decode<Context> for Profile {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let (level, depth) = Decode::decode(decoder)?;
        Self::new(level, depth).ok_or(DecodeError::Other("a profile is level 3 to 5, keeping 1 to 8192 hits"))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Profile {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Decode::decode(decoder)
    }
}

/// Each chunk's TUC profile hits in chunk order: what a profiled launch replies with, and nothing
/// for an unprofiled one. A hit holds its marker id in the low 14 bits and the TUC cycle count from
/// bit 16 up.
pub type Records = Vec<Vec<u64>>;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::Answer;
    use crate::frame::{self, MAX_RESPONSE_WORDS};

    /// A profile is level 3 to 5 and keeps 1 to `QUEUE` hits.
    #[test]
    fn refuses_profiles_the_firmware_cannot_record() {
        assert!(Profile::new(3, 1).is_some() && Profile::new(5, Profile::QUEUE).is_some());
        for (level, depth) in [(2, 1), (6, 1), (3, 0), (5, Profile::QUEUE + 1)] {
            assert_eq!(Profile::new(level, depth), None, "level {level}, depth {depth}");
        }
    }

    /// Every profile a launch fits answers in one response, and the fullest fills it exactly.
    #[test]
    fn answers_a_full_profile_in_one_response() {
        let depth = Profile::HITS / Profile::CHUNKS;
        let profile = Profile::new(5, depth).unwrap();
        assert!(profile.fits(Profile::CHUNKS as usize));
        assert!(!profile.fits(Profile::CHUNKS as usize + 1));
        assert!(!Profile::new(5, depth + 1).unwrap().fits(Profile::CHUNKS as usize));
        let records: Records = (0..u64::from(Profile::CHUNKS))
            .map(|chunk| vec![chunk; depth as usize])
            .collect();
        let mut frame = vec![0; MAX_RESPONSE_WORDS];
        let len = frame::encode(&Answer(Ok(records.clone())), &mut frame).unwrap();
        assert_eq!(len, MAX_RESPONSE_WORDS);
        assert_eq!(frame::decode::<Answer<Records>>(&frame[..len]), Ok(Answer(Ok(records))));
    }
}
