//! How messages lie on the wire: fixed-width little-endian frames with aligned, borrowed launch
//! arguments.

use bincode::error::{DecodeError, EncodeError};

use crate::{Answer, Profile, Request};

/// Encodes one message into caller-owned words, returning the frame length in words; a frame past
/// the message's [`Message::MAX_WORDS`] is refused, however much room `out` has.
pub fn encode<T: Message>(message: &T, out: &mut [u64]) -> Result<usize, Error> {
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(out);
    let (header, payload) = bytes.split_at_mut_checked(HEADER_BYTES).ok_or(Error::Capacity)?;
    let len = bincode::encode_into_slice(message, payload, config()).map_err(|error| match error {
        EncodeError::Other(message) => Error::Encode(message),
        _ => Error::Capacity,
    })?;
    let words = (HEADER_BYTES + len).div_ceil(WORD_BYTES);
    if words > T::MAX_WORDS {
        return Err(Error::Capacity);
    }
    payload[len..words * WORD_BYTES - HEADER_BYTES].fill(0);
    let version = VERSION;
    let words = u16::try_from(words).map_err(|_| Error::Capacity)?;
    bincode::encode_into_slice(Header { version, words }, header, config()).map_err(|_| Error::Capacity)?;
    Ok(usize::from(words))
}

/// Decodes the frame `words` open with, borrowing inline launch arguments from them. Words past the
/// frame are padding; [`Error::Truncated`] says the frame has not all arrived.
pub fn decode<'a, T: bincode::BorrowDecode<'a, ()>>(words: &'a [u64]) -> Result<T, Error> {
    let header = Header::read(*words.first().ok_or(Error::Truncated)?)?;
    let frame: &[u8] = bytemuck::cast_slice(words.get(..usize::from(header.words)).ok_or(Error::Truncated)?);
    let bytes: &[u8] = bytemuck::cast_slice(words);
    let (message, used) =
        bincode::borrow_decode_from_slice(&frame[HEADER_BYTES..], config()).map_err(|error| match error {
            DecodeError::Other(message) => Error::Malformed(message),
            _ => Error::Decode,
        })?;
    if bytes[HEADER_BYTES + used..].iter().any(|byte| *byte != 0) {
        return Err(Error::Padding);
    }
    Ok(message)
}

/// A message one side sends, and the most words its frame takes, this header included.
pub trait Message: bincode::Encode {
    const MAX_WORDS: usize;
}

impl Message for Request<'_> {
    const MAX_WORDS: usize = MAX_SUBMISSION_WORDS;
}

impl<T: bincode::Encode> Message for Answer<T> {
    const MAX_WORDS: usize = MAX_RESPONSE_WORDS;
}

#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Frame has not fully arrived.
    #[error("frame needs more words than it has")]
    Truncated,
    /// Frame was written against another ABI revision.
    #[error("frame speaks ABI version {0}, expected {VERSION}")]
    Version(u16),
    /// Encoded frame exceeds the buffer or the peer's frame limit.
    #[error("encoded frame exceeds capacity")]
    Capacity,
    /// The message's encoder rejected its data.
    #[error("cannot encode message: {0}")]
    Encode(&'static str),
    /// The message's decoder rejected its data.
    #[error("invalid message: {0}")]
    Malformed(&'static str),
    /// The payload cannot be decoded under the fixed-integer encoding.
    #[error("frame contains an invalid bincode payload")]
    Decode,
    /// Bytes after the decoded payload must be zero.
    #[error("frame padding is nonzero")]
    Padding,
}

/// A frame's revision and length in words, including this four-byte header.
#[derive(bincode::Encode, bincode::Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    version: u16,
    words: u16,
}

impl Header {
    /// Whether every word of the frame `words` open with has arrived; a header that cannot be read
    /// counts as arrived, so decoding reports why.
    pub fn arrived(words: &[u64]) -> bool {
        words
            .first()
            .is_some_and(|&word| Self::read(word).map_or(true, |header| words.len() >= usize::from(header.words)))
    }

    /// The header in a frame's first word, if it speaks this ABI at a length either side accepts.
    fn read(word: u64) -> Result<Self, Error> {
        let (header, _) = bincode::decode_from_slice::<Self, _>(&word.to_ne_bytes()[..HEADER_BYTES], config())
            .map_err(|_| Error::Decode)?;
        if header.version != VERSION {
            return Err(Error::Version(header.version));
        }
        if !(1..=MAX_RESPONSE_WORDS).contains(&usize::from(header.words)) {
            return Err(Error::Decode);
        }
        Ok(header)
    }
}

/// Maximum payload words after the kernel submission entry's ring header.
pub const MAX_SUBMISSION_WORDS: usize = 1008 / size_of::<u64>() - 1;
/// Maximum response frame: a failure with its whole diagnostic string, or a launch's reply with
/// every hit its profile may record.
pub const MAX_RESPONSE_WORDS: usize = {
    let failure = HEADER_BYTES + size_of::<i32>() + size_of::<u64>() + DIAGNOSTIC_BYTES;
    // The chunk count, then each chunk's hit count and hits.
    let records = size_of::<u64>() * (1 + Profile::CHUNKS as usize + Profile::HITS as usize);
    let launch = HEADER_BYTES + size_of::<i32>() + records;
    (if failure > launch { failure } else { launch }).div_ceil(size_of::<u64>())
};
/// Diagnostic byte budget; the status and string length are additional frame bytes.
pub const DIAGNOSTIC_BYTES: usize = 8192 * size_of::<u64>();

/// Host and firmware reject differing revisions. Only a change of the header takes a new one: calls
/// and options come and go by their tags, and a released tag's payload never changes, so a changed
/// payload takes a new variant name. The golden frames of `preserves_*_frames` pin each one.
pub(crate) const VERSION: u16 = 1;

pub(crate) const HEADER_BYTES: usize = size_of::<Header>();
const WORD_BYTES: usize = size_of::<u64>();
const _: () = assert!(HEADER_BYTES == 4);
const _: () = assert!(MAX_RESPONSE_WORDS <= u16::MAX as usize);
const _: () = assert!(cfg!(target_endian = "little"), "frames are native little-endian words");

/// Fixed-width little-endian integers; the bincode limit applies only while decoding.
const fn config() -> bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
    bincode::config::Limit<{ MAX_RESPONSE_WORDS * WORD_BYTES }>,
> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
        .with_limit()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::message::MAX_LAUNCH_ARGS;
    use crate::{Answer, Args, Configure, ConfigureOption, Failure, Identity, Launch, Request, Staged};

    use super::*;

    /// Encoding never creates a frame beyond the peer's accepted word budget.
    #[test]
    fn refuses_oversized_responses() {
        let failure = Failure {
            status: 9,
            message: "x".repeat(MAX_RESPONSE_WORDS * WORD_BYTES),
        };
        let mut out = vec![0; MAX_RESPONSE_WORDS + 8];
        assert_eq!(encode(&Answer::<()>(Err(failure)), &mut out), Err(Error::Capacity));
    }

    /// A frame stays within its message's limit, however much room the caller gives it.
    #[test]
    fn bounds_frames_by_their_message() {
        /// One word more than a submission frame carries.
        struct Oversized;

        impl bincode::Encode for Oversized {
            fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
                bincode::Encode::encode(&[0u64; MAX_SUBMISSION_WORDS], encoder)
            }
        }

        impl Message for Oversized {
            const MAX_WORDS: usize = MAX_SUBMISSION_WORDS;
        }

        assert_eq!(encode(&Oversized, &mut [0; MAX_RESPONSE_WORDS]), Err(Error::Capacity));
    }

    /// A permanent argument-count rejection is not a request to wait for more words.
    #[test]
    fn distinguishes_invalid_arguments_from_incomplete_frames() {
        assert_eq!(
            encode(&launch(&[0; MAX_LAUNCH_ARGS + 1]), &mut [0; MAX_SUBMISSION_WORDS + 8]),
            Err(Error::Encode("too many launch arguments")),
        );
    }

    fn words<T: Message>(message: &T, limit: usize) -> Vec<u64> {
        let mut out = vec![0; limit];
        let len = encode(message, &mut out).expect("message fits");
        out.truncate(len);
        out
    }

    fn launch(args: &[u64]) -> Request<'_> {
        Request::Launch(Launch {
            args: Args::Inline(args),
            executable: Staged {
                addr: 0xc0_0000_1000,
                len: 44_000,
                token: 3,
            },
            options: Vec::new(),
        })
    }

    fn configure() -> Request<'static> {
        Request::Configure(Configure {
            options: vec![
                ConfigureOption::Identity(Identity {
                    physical: 9,
                    rank: 1,
                    chips: 2,
                    cluster: 3,
                }),
                ConfigureOption::LogLevel(4),
            ],
        })
    }

    /// A frame from another ABI revision is not interpreted as a current message.
    #[test]
    fn rejects_other_versions() {
        let mut version = words(&configure(), MAX_SUBMISSION_WORDS);
        version[0] ^= 1;
        assert_eq!(decode::<Request<'_>>(&version), Err(Error::Version(VERSION ^ 1)));
    }

    /// Extra words are accepted only when all their bytes are zero.
    #[test]
    fn validates_trailing_words() {
        for (word, expected) in [(0, Ok(configure())), (1, Err(Error::Padding))] {
            let mut frame = words(&configure(), MAX_SUBMISSION_WORDS);
            frame.push(word);
            assert_eq!(decode::<Request<'_>>(&frame), expected);
        }
    }

    /// Unused bytes inside the declared final word must also be zero.
    #[test]
    fn rejects_nonzero_padding() {
        let failure = Failure {
            status: 1,
            message: "x".into(),
        };
        let mut padding = words(&Answer::<()>(Err(failure)), MAX_RESPONSE_WORDS);
        let last = padding.len() - 1;
        padding[last] |= 1 << 56;
        assert_eq!(decode::<Answer<()>>(&padding), Err(Error::Padding));
    }

    /// A missing word is incomplete rather than a permanently malformed payload.
    #[test]
    fn reports_incomplete_frames() {
        let mut short = words(&launch(&[1, 2]), MAX_SUBMISSION_WORDS);
        short.pop();
        assert_eq!(decode::<Request<'_>>(&short), Err(Error::Truncated));
    }

    /// Declared frame lengths must fit the peer's accepted range.
    #[test]
    fn rejects_invalid_lengths() {
        for header in [u64::from(VERSION), 0xffff_0000 | u64::from(VERSION)] {
            assert_eq!(decode::<Answer<()>>(&[header]), Err(Error::Decode));
        }
    }
}
