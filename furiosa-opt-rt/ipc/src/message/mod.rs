use alloc::string::String;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use core::ops::Range;
use furiosa_opt_abi::arch::rngd::dma::Config;

mod request;
pub(crate) use request::MAX_LAUNCH_ARGS;

/// Device code and its generation identity. `token` distinguishes address reuse.
#[derive(Decode, Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staged {
    /// Device-space byte address of the staged image.
    pub addr: u64,
    /// Length of the staged image in bytes.
    pub len: u64,
    pub token: u64,
}

impl Staged {
    /// Validates the code bytes against the consumer's device address space.
    pub fn range(&self, memory: Range<u64>) -> Option<Range<usize>> {
        within(memory, self.addr, self.len)
    }
}

/// Host to device request. A launch's arguments come first in the frame, where they sit on a
/// word boundary and the device reads them in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request<'a> {
    /// Runs the executable with the supplied arguments.
    Launch {
        /// Argument words, inline or in a device-memory table retained through completion.
        args: Args<'a>,
        /// Staged device image and its generation token.
        executable: Staged,
    },
    /// Gives the cluster its identity for selecting task code from an image.
    Initialize {
        /// Hardware identity reported by the management driver.
        physical: u8,
        /// This chip's index among the device's `chips`.
        rank: u8,
        /// Number of chips in the device.
        chips: u8,
        /// Cluster index in the device's executable image.
        cluster: u8,
        /// `log::LevelFilter as usize`, encoded in one byte (0 off, 5 trace).
        log_level: u8,
    },
    /// Samples the cluster ARM counter without executing compiler code.
    Clock,
    /// Assigns kernel-reserved PCIe DMA registers, descriptor storage and channels to firmware.
    Dma(Config),
    /// Copies `len` bytes on the cluster's DMA channels between a host range mapped for this
    /// cluster, as `furiosa_opt_abi::arch::rngd::dma::Operand` encodes it, and its local memory.
    Copy { source: u64, destination: u64, len: u64 },
}

/// Arguments borrowed from this frame or from device memory for one launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Args<'a> {
    Inline(&'a [u64]),
    Indirect { address: u64, count: u32 },
}

impl Args<'_> {
    /// Whether `count` arguments fit in a launch frame; more go [`Args::Indirect`].
    pub const fn fits_inline(count: usize) -> bool {
        count <= MAX_LAUNCH_ARGS
    }

    /// Indirect arguments must be u64-aligned for the firmware's borrowed word view.
    pub fn range(&self, memory: Range<u64>) -> Option<Range<usize>> {
        let Self::Indirect { address, count } = *self else {
            return None;
        };
        if !address.is_multiple_of(size_of::<u64>() as u64) {
            return None;
        }
        within(memory, address, u64::from(count) * size_of::<u64>() as u64)
    }
}

fn within(memory: Range<u64>, start: u64, len: u64) -> Option<Range<usize>> {
    let end = start.checked_add(len)?;
    if len == 0 || start < memory.start || end > memory.end {
        return None;
    }
    Some(usize::try_from(start).ok()?..usize::try_from(end).ok()?)
}

/// Device to host response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Response {
    /// A request failed with a [`Status`](crate::Status) value, possibly one this side does not name,
    /// and whatever detail the firmware has, such as a panicked task's assertion.
    Error { status: i32, message: String },
    /// The cluster took its identity.
    Initialize,
    /// Every task of a launch exited cleanly.
    Launch,
    /// Samples the ARM generic counter in ticks at the platform counter frequency.
    Clock { counter: u64 },
    /// The cluster accepted its PCIe DMA configuration.
    Dma,
    /// Every byte of a copy landed.
    Copy,
}

impl Encode for Response {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        match self {
            Self::Error { status, message } => (0u32, status, message).encode(encoder),
            Self::Initialize => 2u32.encode(encoder),
            Self::Launch => 3u32.encode(encoder),
            Self::Clock { counter } => (4u32, counter).encode(encoder),
            Self::Dma => 5u32.encode(encoder),
            Self::Copy => 6u32.encode(encoder),
        }
    }
}

impl<Context> Decode<Context> for Response {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        match u32::decode(decoder)? {
            0 => {
                let (status, message) = Decode::decode(decoder)?;
                Ok(Self::Error { status, message })
            }
            2 => Ok(Self::Initialize),
            3 => Ok(Self::Launch),
            4 => Ok(Self::Clock {
                counter: Decode::decode(decoder)?,
            }),
            5 => Ok(Self::Dma),
            6 => Ok(Self::Copy),
            _ => Err(DecodeError::Other("unknown response")),
        }
    }
}

bincode::impl_borrow_decode!(Response);

#[cfg(test)]
mod tests {
    use super::{Args, Response, Staged};
    use crate::wire::{self, Error, MAX_RESPONSE_WORDS};

    /// Revision-four golden frames pin each response tag and fixed-width field order.
    #[test]
    fn preserves_response_frames() {
        let cases: [(Response, &[u64]); 6] = [
            (
                Response::Error {
                    status: -7,
                    message: "xy".into(),
                },
                &[0x0003_0004, 0x0000_0002_ffff_fff9, 0x0000_7978_0000_0000],
            ),
            (Response::Initialize, &[0x0000_0002_0001_0004]),
            (Response::Launch, &[0x0000_0003_0001_0004]),
            (Response::Clock { counter: 37 }, &[0x0000_0004_0002_0004, 37]),
            (Response::Dma, &[0x0000_0005_0001_0004]),
            (Response::Copy, &[0x0000_0006_0001_0004]),
        ];
        for (response, golden) in cases {
            let mut frame = alloc::vec![0; MAX_RESPONSE_WORDS];
            let len = wire::encode(&response, &mut frame).unwrap();
            assert_eq!(&frame[..len], golden);
            assert_eq!(wire::decode::<Response>(golden), Ok(response));
        }
    }

    /// Retired response tags cannot be interpreted as another answer.
    #[test]
    fn refuses_unknown_responses() {
        assert!(matches!(
            wire::decode::<Response>(&[0x0000_0001_0001_0004]),
            Err(Error::Malformed(_))
        ));
    }

    /// The firmware takes host-given code and argument ranges only inside device memory.
    #[test]
    fn bounds_device_ranges() {
        let executable = |addr, len| Staged { addr, len, token: 0 }.range(4096..8192);
        assert_eq!(executable(4096, 4096), Some(4096..8192));
        assert!(executable(4096, 0).is_none() && executable(8191, 2).is_none());
        assert_eq!(
            Args::Indirect {
                address: 4096,
                count: 512
            }
            .range(4096..8192),
            Some(4096..8192)
        );
        for (address, count) in [
            (4096, 0),
            (4097, 1),
            (4088, 1),
            (8192, 1),
            (4096, 513),
            (4096, u32::MAX),
        ] {
            assert!(Args::Indirect { address, count }.range(4096..8192).is_none());
        }
        assert!(
            Args::Indirect {
                address: u64::MAX - 7,
                count: 1
            }
            .range(0..u64::MAX)
            .is_none()
        );
    }
}
