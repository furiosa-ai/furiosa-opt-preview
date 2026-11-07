//! A launch's TUC profile: what the request asks every chunk to record, and what the reply carries
//! back of each chunk.

use alloc::vec::Vec;
use bincode::de::{BorrowDecoder, Decoder};
use bincode::error::DecodeError;
use bincode::{BorrowDecode, Decode, Encode};

/// Profiles a launch at the task library's `level`, `Info` (3) through `Trace` (5), recording the
/// markers `selection` holds besides the Task's. Exists only in that level range.
#[derive(Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    level: u8,
    selection: Selection,
}

impl Profile {
    /// Hits the firmware's TUC profile queue holds, every chunk's own.
    pub const QUEUE: u32 = 8192;
    /// Hits one reply carries, every chunk's together.
    pub const HITS: u32 = 8192;
    /// Chunks one reply carries the hits of.
    pub const CHUNKS: u32 = 256;

    /// A profile at `level`, or `None` outside the levels the firmware knows.
    pub fn new(level: u8, selection: Selection) -> Option<Self> {
        (3..=5).contains(&level).then_some(Self { level, selection })
    }

    pub fn level(self) -> u8 {
        self.level
    }

    pub fn selection(self) -> Selection {
        self.selection
    }
}

impl<Context> Decode<Context> for Profile {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let (level, selection) = Decode::decode(decoder)?;
        Self::new(level, selection).ok_or(DecodeError::Other("a profile is level 3 to 5"))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Profile {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Decode::decode(decoder)
    }
}

/// The markers a launch records besides the Task's, as the device code tests them: ids
/// `low..high`, while the loop whose trip marker is `trip` runs one of its trips
/// `first..=last`. The default selects every marker.
#[derive(Encode, Decode, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Selection {
    /// `low..high` in the low and high halves; zero for every id.
    pub range: u32,
    /// The selected loop's trip marker; zero for none, every trip.
    pub trip: u16,
    pub first: u32,
    pub last: u32,
}

/// One chunk's profile: the hits the reply carries, oldest first, and how many the chunk's code
/// tried to record. A hit holds its marker id in the low 16 bits and the TUC cycle count from bit
/// 16 up.
#[derive(Encode, Decode, Debug, Clone, Default, PartialEq, Eq)]
pub struct Chunk {
    pub attempted: u32,
    pub hits: Vec<u64>,
}

/// Each chunk's profile in chunk order: what a profiled launch replies with, and nothing for an
/// unprofiled one.
pub type Records = Vec<Chunk>;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::Answer;
    use crate::frame::{self, MAX_RESPONSE_WORDS};

    /// A profile is level 3 to 5.
    #[test]
    fn refuses_profiles_the_firmware_cannot_record() {
        assert!(Profile::new(3, Selection::default()).is_some());
        assert!(Profile::new(5, Selection::default()).is_some());
        for level in [2, 6] {
            assert_eq!(Profile::new(level, Selection::default()), None, "level {level}");
        }
    }

    /// The fullest reply, every chunk with its share of the hits one reply carries, answers in one
    /// response.
    #[test]
    fn answers_a_full_profile_in_one_response() {
        let depth = Profile::HITS / Profile::CHUNKS;
        let records: Records = (0..u64::from(Profile::CHUNKS))
            .map(|chunk| Chunk {
                attempted: u32::MAX,
                hits: vec![chunk; depth as usize],
            })
            .collect();
        let mut frame = vec![0; MAX_RESPONSE_WORDS];
        let len = frame::encode(&Answer(Ok(records.clone())), &mut frame).unwrap();
        assert_eq!(len, MAX_RESPONSE_WORDS);
        assert_eq!(frame::decode::<Answer<Records>>(&frame[..len]), Ok(Answer(Ok(records))));
    }
}
