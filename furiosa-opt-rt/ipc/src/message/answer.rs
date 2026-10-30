//! What a cluster answers a request with: the reply, or why it refused the request.

use alloc::string::String;
use bincode::de::{BorrowDecoder, Decoder};
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};

use crate::Status;

/// A request's answer as a frame carries it: the status, zero for success, then the reply or the
/// failure's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer<T>(pub Result<T, Failure>);

/// A refused request: a [`Status`] value, possibly one this side does not name, and whatever detail
/// the firmware has, such as a panicked task's assertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub status: i32,
    pub message: String,
}

impl<T: Encode> Encode for Answer<T> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        match &self.0 {
            Ok(reply) => (0i32, reply).encode(encoder),
            Err(Failure { status: 0, .. }) => Err(EncodeError::Other("a failure needs a nonzero status")),
            Err(Failure { status, message }) => (status, message).encode(encoder),
        }
    }
}

impl<Context, T: Decode<Context>> Decode<Context> for Answer<T> {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self(match i32::decode(decoder)? {
            0 => Ok(T::decode(decoder)?),
            status => Err(Failure {
                status,
                message: String::decode(decoder)?,
            }),
        }))
    }
}

impl<'de, Context, T: Decode<Context>> BorrowDecode<'de, Context> for Answer<T> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Self::decode(decoder)
    }
}

impl Failure {
    /// A refusal with `status`, its message cut to what one answer frame carries.
    pub fn new(status: Status, message: impl Into<String>) -> Self {
        let mut message = message.into();
        message.truncate(message.floor_char_boundary(crate::frame::DIAGNOSTIC_BYTES));
        Self {
            status: status.into(),
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Records;
    use crate::frame::{self, MAX_RESPONSE_WORDS};

    /// Revision-one golden frames pin the status word and each reply's field order.
    #[test]
    fn preserves_answer_frames() {
        let failure = Failure {
            status: -7,
            message: "xy".into(),
        };
        let records: Records = alloc::vec![alloc::vec![1 | 100 << 16], alloc::vec![]];
        let encoded = |encode: &dyn Fn(&mut [u64]) -> Result<usize, frame::Error>| {
            let mut frame = alloc::vec![0; MAX_RESPONSE_WORDS];
            let len = encode(&mut frame).unwrap();
            frame.truncate(len);
            frame
        };
        assert_eq!(
            encoded(&|frame| frame::encode(&Answer::<()>(Ok(())), frame)),
            [0x0000_0000_0001_0001]
        );
        assert_eq!(
            encoded(&|frame| frame::encode(&Answer(Ok(37_u64)), frame)),
            [0x0000_0000_0002_0001, 37]
        );
        assert_eq!(
            encoded(&|frame| frame::encode(&Answer(Ok(records.clone())), frame)),
            [0x0000_0000_0005_0001, 2, 1, 0x0064_0001, 0]
        );
        assert_eq!(
            encoded(&|frame| frame::encode(&Answer::<()>(Err(failure.clone())), frame)),
            [0xffff_fff9_0003_0001, 2, 0x7978]
        );
        assert_eq!(
            frame::decode::<Answer<()>>(&[0x0000_0000_0001_0001]),
            Ok(Answer(Ok(())))
        );
        assert_eq!(
            frame::decode::<Answer<u64>>(&[0x0000_0000_0002_0001, 37]),
            Ok(Answer(Ok(37)))
        );
        assert_eq!(
            frame::decode::<Answer<Records>>(&[0x0000_0000_0005_0001, 2, 1, 0x0064_0001, 0]),
            Ok(Answer(Ok(records)))
        );
        assert_eq!(
            frame::decode::<Answer<()>>(&[0xffff_fff9_0003_0001, 2, 0x7978]),
            Ok(Answer(Err(failure)))
        );
    }

    /// A failure always carries a status, so it cannot read as success.
    #[test]
    fn refuses_failures_without_status() {
        let failure = Failure {
            status: 0,
            message: String::new(),
        };
        let mut frame = alloc::vec![0; MAX_RESPONSE_WORDS];
        assert!(frame::encode(&Answer::<()>(Err(failure)), &mut frame).is_err());
    }

    /// A message past the diagnostic budget is cut at a character boundary.
    #[test]
    fn bounds_failure_messages() {
        let failure = Failure::new(Status::Panicked, "é".repeat(frame::DIAGNOSTIC_BYTES));
        assert!(failure.message.len() <= frame::DIAGNOSTIC_BYTES);
        let mut frame = alloc::vec![0; MAX_RESPONSE_WORDS];
        assert!(frame::encode(&Answer::<()>(Err(failure)), &mut frame).is_ok());
    }
}
