use bincode::de::BorrowDecoder;
use bincode::de::read::BorrowReader;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};

use super::{Args, Request, Staged};
use crate::wire::{HEADER_BYTES, MAX_SUBMISSION_WORDS};

const LAUNCH_ARGS_AT: usize = HEADER_BYTES + size_of::<u32>() + size_of::<u64>();
const _: () = assert!(LAUNCH_ARGS_AT.is_multiple_of(size_of::<u64>()));
// Fixed-integer encoding writes the three staged-image u64 fields without padding.
const _: () = assert!(size_of::<Staged>() == 3 * size_of::<u64>());
pub(crate) const MAX_LAUNCH_ARGS: usize =
    (MAX_SUBMISSION_WORDS * size_of::<u64>() - LAUNCH_ARGS_AT - size_of::<Staged>()) / size_of::<u64>();

// Host and firmware use these explicit wire tags; removed tags remain reserved.
impl Encode for Request<'_> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        match self {
            Self::Launch {
                args: Args::Inline(args),
                executable,
            } => {
                if !Args::fits_inline(args.len()) {
                    return Err(EncodeError::Other("too many launch arguments"));
                }
                (2u32, args, executable).encode(encoder)
            }
            Self::Launch {
                args: Args::Indirect { address, count },
                executable,
            } => (6u32, address, count, executable).encode(encoder),
            Self::Initialize {
                physical,
                rank,
                chips,
                cluster,
                log_level,
            } => (3u32, physical, rank, chips, cluster, log_level).encode(encoder),
            Self::Clock => 4u32.encode(encoder),
            Self::Dma(config) => (5u32, config).encode(encoder),
            Self::Copy {
                source,
                destination,
                len,
            } => (7u32, source, destination, len).encode(encoder),
        }
    }
}

impl<'a, 'de: 'a, Context> BorrowDecode<'de, Context> for Request<'a> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        match u32::decode(decoder)? {
            2 => {
                let count = u64::decode(decoder)?;
                if count > MAX_LAUNCH_ARGS as u64 {
                    return Err(DecodeError::Other("too many launch arguments"));
                }
                let bytes = decoder.borrow_reader().take_bytes(count as usize * size_of::<u64>())?;
                let args = bytemuck::try_cast_slice(bytes)
                    .map_err(|_| DecodeError::Other("launch arguments are not word aligned"))?;
                Ok(Self::Launch {
                    args: Args::Inline(args),
                    executable: Decode::decode(decoder)?,
                })
            }
            3 => {
                let (physical, rank, chips, cluster, log_level) = Decode::decode(decoder)?;
                Ok(Self::Initialize {
                    physical,
                    rank,
                    chips,
                    cluster,
                    log_level,
                })
            }
            4 => Ok(Self::Clock),
            5 => Ok(Self::Dma(Decode::decode(decoder)?)),
            6 => {
                let (address, count, executable) = Decode::decode(decoder)?;
                Ok(Self::Launch {
                    args: Args::Indirect { address, count },
                    executable,
                })
            }
            7 => {
                let (source, destination, len) = Decode::decode(decoder)?;
                Ok(Self::Copy {
                    source,
                    destination,
                    len,
                })
            }
            _ => Err(DecodeError::Other("unknown request")),
        }
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_abi::arch::rngd::dma::Config;

    use super::*;
    use crate::wire::{self, Error};

    /// Revision-four golden frames pin every request tag, including both launch argument modes.
    #[test]
    fn preserves_request_frames() {
        let staged = Staged {
            addr: 17,
            len: 19,
            token: 23,
        };
        let cases: [(Request<'_>, &[u64]); 6] = [
            (
                Request::Launch {
                    args: Args::Inline(&[41, 43]),
                    executable: staged,
                },
                &[0x0000_0002_0007_0004, 2, 41, 43, 17, 19, 23],
            ),
            (
                Request::Initialize {
                    physical: 9,
                    rank: 1,
                    chips: 2,
                    cluster: 3,
                    log_level: 4,
                },
                &[0x0000_0003_0002_0004, 0x0000_0004_0302_0109],
            ),
            (Request::Clock, &[0x0000_0004_0001_0004]),
            (
                Request::Dma(Config {
                    registers: 11,
                    descriptors: 13,
                    channels: 17,
                    local: 19,
                    peers: [23, 29, 31, 37, 41, 43, 47, 53],
                    members: 8,
                }),
                &[
                    0x0000_0005_000d_0004,
                    11,
                    13,
                    0x1311,
                    0x1700,
                    0x1d00,
                    0x1f00,
                    0x2500,
                    0x2900,
                    0x2b00,
                    0x2f00,
                    0x3500,
                    0x0800,
                ],
            ),
            (
                Request::Launch {
                    args: Args::Indirect {
                        address: 31,
                        count: 4097,
                    },
                    executable: staged,
                },
                &[
                    0x0000_0006_0006_0004,
                    31,
                    0x0000_0011_0000_1001,
                    0x0000_0013_0000_0000,
                    0x0000_0017_0000_0000,
                    0,
                ],
            ),
            (
                Request::Copy {
                    source: 29,
                    destination: 31,
                    len: 37,
                },
                &[0x0000_0007_0004_0004, 29, 31, 37],
            ),
        ];
        for (request, golden) in cases {
            let mut frame = [0; MAX_SUBMISSION_WORDS];
            let len = wire::encode(&request, &mut frame).unwrap();
            assert_eq!(&frame[..len], golden);
            assert_eq!(wire::decode::<Request<'_>>(golden), Ok(request));
        }
    }

    /// Retired tags and over-limit inline counts are rejected before argument borrowing.
    #[test]
    fn refuses_unknown_or_oversized_requests() {
        assert!(matches!(
            wire::decode::<Request<'_>>(&[0x0000_0001_0001_0004]),
            Err(Error::Malformed(_))
        ));
        assert!(matches!(
            wire::decode::<Request<'_>>(&[0x0000_0002_0002_0004, MAX_LAUNCH_ARGS as u64 + 1]),
            Err(Error::Malformed(_))
        ));
    }

    /// The submission entry bounds inline arguments; larger tables retain indirect addressing.
    #[test]
    fn preserves_argument_frames() {
        let inline = core::array::from_fn::<_, MAX_LAUNCH_ARGS, _>(|index| index as u64 + 7);
        let staged = Staged {
            addr: 0xc0_0000_1000,
            len: 512,
            token: 7,
        };
        for count in [0, 1, MAX_LAUNCH_ARGS, MAX_LAUNCH_ARGS + 1, 4097] {
            let args = match count {
                ..=MAX_LAUNCH_ARGS => Args::Inline(&inline[..count]),
                _ => Args::Indirect {
                    address: 0xc0_0000_2000,
                    count: count as u32,
                },
            };
            let request = Request::Launch {
                args,
                executable: staged,
            };
            let mut frame = [0; MAX_SUBMISSION_WORDS];
            let len = wire::encode(&request, &mut frame).unwrap();
            assert_eq!(wire::decode::<Request<'_>>(&frame[..len]).unwrap(), request);
            assert_eq!(len == MAX_SUBMISSION_WORDS, count == MAX_LAUNCH_ARGS);
        }
    }
}
