//! How a launch lies on the wire: its arguments first, so inline words start on a word boundary
//! of the frame, then its executable and its options.

use bincode::de::BorrowDecoder;
use bincode::de::read::BorrowReader;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};

use super::{Args, Launch, Staged, tag};
use crate::frame::{HEADER_BYTES, MAX_SUBMISSION_WORDS};

/// The most argument words a launch frame carries inline; more go in a device-memory table.
pub const MAX_LAUNCH_ARGS: usize =
    (MAX_SUBMISSION_WORDS * size_of::<u64>() - LAUNCH_ARGS_AT - size_of::<Staged>() - OPTIONS_BYTES) / size_of::<u64>();

impl Encode for Launch<'_> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        (&self.args, &self.executable).encode(encoder)?;
        tag::encode_options(&self.options, encoder)
    }
}

impl<'a, 'de: 'a, Context> BorrowDecode<'de, Context> for Launch<'a> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            args: BorrowDecode::borrow_decode(decoder)?,
            executable: Decode::decode(decoder)?,
            options: tag::decode_options(decoder)?,
        })
    }
}

// Arguments are tagged by hand: a four-byte tag and count put the inline words on a word boundary.
const INLINE: u32 = tag::of("Inline");
const INDIRECT: u32 = tag::of("Indirect");

impl Args<'_> {
    /// Every variant's tag, in declaration order.
    pub(crate) const TAGS: &'static [u32] = &[INLINE, INDIRECT];
}

const _: () = assert!(tag::distinct(Args::TAGS), "two Args variants share a tag");

/// The variant's tag and the argument count, then the inline words or the table's address.
impl Encode for Args<'_> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        match *self {
            Self::Inline(args) if args.len() > MAX_LAUNCH_ARGS => Err(EncodeError::Other("too many launch arguments")),
            Self::Inline(args) => {
                (INLINE, args.len() as u32).encode(encoder)?;
                encoder.writer().write(bytemuck::cast_slice(args))
            }
            Self::Indirect { address, count } => (INDIRECT, count, address).encode(encoder),
        }
    }
}

impl<'a, 'de: 'a, Context> BorrowDecode<'de, Context> for Args<'a> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let (tag, count): (u32, u32) = Decode::decode(decoder)?;
        match tag {
            INLINE if count as usize > MAX_LAUNCH_ARGS => Err(DecodeError::Other("too many launch arguments")),
            INLINE => {
                let bytes = decoder.borrow_reader().take_bytes(count as usize * size_of::<u64>())?;
                bytemuck::try_cast_slice(bytes)
                    .map(Self::Inline)
                    .map_err(|_| DecodeError::Other("launch arguments are not word aligned"))
            }
            INDIRECT => Ok(Self::Indirect {
                address: Decode::decode(decoder)?,
                count,
            }),
            _ => Err(DecodeError::Other("unknown Args")),
        }
    }
}

/// Where inline arguments start: after the header, the call's tag, and the arguments' tag and count.
const LAUNCH_ARGS_AT: usize = HEADER_BYTES + 3 * size_of::<u32>();
const _: () = assert!(LAUNCH_ARGS_AT.is_multiple_of(size_of::<u64>()));
// Fixed-integer encoding writes the three staged-image u64 fields without padding.
const _: () = assert!(size_of::<Staged>() == 3 * size_of::<u64>());
/// A launch's options at their largest: the count, then the profile's tag, level and depth.
const OPTIONS_BYTES: usize = size_of::<u32>() + size_of::<u32>() + size_of::<u8>() + size_of::<u32>();

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::frame::{self, Error};
    use crate::{Clock, Copy, LaunchOption, Profile, Request};

    fn launch(args: Args<'_>, options: Vec<LaunchOption>) -> Request<'_> {
        Request::Launch(Launch {
            args,
            executable: Staged {
                addr: 17,
                len: 19,
                token: 23,
            },
            options,
        })
    }

    /// The profiled launch of `preserves_request_frames`.
    const PROFILED: [u64; 9] = [
        0x37cb_b286_0009_0001,
        0x0000_0002_ef92_1554,
        41,
        43,
        17,
        19,
        23,
        0xcd17_328e_0000_0001,
        0x0000_0000_0000_1004,
    ];

    /// Revision-one golden frames pin every call's tag and fields, both argument modes, and a launch
    /// with and without the profile option.
    #[test]
    fn preserves_request_frames() {
        let profile = vec![LaunchOption::Profile(Profile::new(4, 16).unwrap())];
        let cases: [(Request<'_>, &[u64]); 5] = [
            (
                launch(Args::Inline(&[41, 43]), vec![]),
                &[0x37cb_b286_0008_0001, 0x0000_0002_ef92_1554, 41, 43, 17, 19, 23, 0],
            ),
            (launch(Args::Inline(&[41, 43]), profile), &PROFILED),
            (
                launch(
                    Args::Indirect {
                        address: 31,
                        count: 4097,
                    },
                    vec![],
                ),
                &[0x37cb_b286_0007_0001, 0x0000_1001_8a3c_0c07, 31, 17, 19, 23, 0],
            ),
            (
                Request::Copy(Copy {
                    source: 29,
                    destination: 31,
                    len: 37,
                }),
                &[0x658f_3664_0004_0001, 29, 31, 37],
            ),
            (Request::Clock(Clock), &[0xbb5f_80fd_0001_0001]),
        ];
        for (request, golden) in cases {
            let mut frame = [0; MAX_SUBMISSION_WORDS];
            let len = frame::encode(&request, &mut frame).unwrap();
            assert_eq!(&frame[..len], golden);
            assert_eq!(frame::decode::<Request<'_>>(golden), Ok(request));
        }
    }

    /// A call this side does not know is refused, never read as another call.
    #[test]
    fn refuses_unknown_calls() {
        let reset = u64::from(tag::of("Reset")) << 32 | 0x0001_0001;
        assert!(matches!(
            frame::decode::<Request<'_>>(&[reset]),
            Err(Error::Malformed(_))
        ));
    }

    /// An option this side does not know is refused rather than skipped.
    #[test]
    fn refuses_unknown_options() {
        let mut frame = PROFILED;
        frame[7] = u64::from(tag::of("Trace")) << 32 | 1;
        assert!(matches!(frame::decode::<Request<'_>>(&frame), Err(Error::Malformed(_))));
    }

    /// Over-limit inline counts and profiles the firmware cannot record are rejected.
    #[test]
    fn refuses_oversized_requests() {
        let frame = [
            0x37cb_b286_0002_0001,
            (MAX_LAUNCH_ARGS as u64 + 1) << 32 | u64::from(INLINE),
        ];
        assert!(matches!(frame::decode::<Request<'_>>(&frame), Err(Error::Malformed(_))));
        // The profiled frame asking for level 6.
        let mut frame = PROFILED;
        frame[8] = 0x0000_0000_0000_1006;
        assert!(matches!(frame::decode::<Request<'_>>(&frame), Err(Error::Malformed(_))));
    }

    /// Launches with and without the profile option round-trip at every argument count. Inline
    /// arguments fill the submission entry only at the maximum with every option; more go indirect.
    #[test]
    fn preserves_argument_frames() {
        let inline = core::array::from_fn::<_, MAX_LAUNCH_ARGS, _>(|index| index as u64 + 7);
        let every = vec![LaunchOption::Profile(Profile::new(5, Profile::QUEUE).unwrap())];
        for options in [vec![], every.clone()] {
            for count in [0, 1, MAX_LAUNCH_ARGS, MAX_LAUNCH_ARGS + 1, 4097] {
                let args = match count {
                    ..=MAX_LAUNCH_ARGS => Args::Inline(&inline[..count]),
                    _ => Args::Indirect {
                        address: 0xc0_0000_2000,
                        count: count as u32,
                    },
                };
                let full = count == MAX_LAUNCH_ARGS && options == every;
                let request = launch(args, options.clone());
                let mut frame = [0; MAX_SUBMISSION_WORDS];
                let len = frame::encode(&request, &mut frame).unwrap();
                assert_eq!(frame::decode::<Request<'_>>(&frame[..len]).unwrap(), request);
                assert_eq!(len == MAX_SUBMISSION_WORDS, full);
            }
        }
    }
}
