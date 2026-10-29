//! The one configuration step. Each of its parts is an option, so a part joins or leaves without
//! changing the frames of the others.

use alloc::vec::Vec;
use bincode::de::BorrowDecoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};
use furiosa_opt_abi::arch::rngd::dma::Config;

use super::tag::{self, tagged};

/// Configures a cluster: the options present, applied together.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Configure {
    /// Each option at most once.
    pub options: Vec<ConfigureOption>,
}

impl Encode for Configure {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        tag::encode_options(&self.options, encoder)
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Configure {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            options: tag::decode_options(decoder)?,
        })
    }
}

/// An option a configuration may carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigureOption {
    Identity(Identity),
    /// The finest `log::Level` the firmware reports, as `log::Level as u8` (0 off, 5 trace).
    LogLevel(u8),
    /// Kernel-reserved PCIe DMA registers, descriptor storage and channels the firmware drives.
    Dma(Config),
    /// How long a launch may run, in milliseconds, before the firmware fails it as timed out.
    LaunchTimeout(u32),
}

tagged! { ConfigureOption { Identity, LogLevel, Dma, LaunchTimeout } }

/// The cluster's place in the device, which selects the task code it runs from an image.
#[derive(Encode, Decode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// Hardware identity reported by the management driver.
    pub physical: u8,
    /// This chip's index among the device's `chips`.
    pub rank: u8,
    /// Number of chips in the device.
    pub chips: u8,
    /// Cluster index in the device's executable image.
    pub cluster: u8,
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::Request;
    use crate::frame::{self, MAX_SUBMISSION_WORDS};

    const IDENTITY: Identity = Identity {
        physical: 9,
        rank: 1,
        chips: 2,
        cluster: 3,
    };

    /// Revision-one golden frames: the option count, then each option's tag and value.
    #[test]
    fn preserves_configuration_frames() {
        let cases: [(Configure, &[u64]); 2] = [
            (
                Configure {
                    options: vec![ConfigureOption::Identity(IDENTITY), ConfigureOption::LogLevel(4)],
                },
                &[0x163e_25eb_0004_0001, 0xdfc1_d0bb_0000_0002, 0x4a33_1979_0302_0109, 4],
            ),
            (Configure::default(), &[0x163e_25eb_0002_0001, 0]),
        ];
        for (configure, golden) in cases {
            let request = Request::Configure(configure);
            let mut frame = [0; MAX_SUBMISSION_WORDS];
            let len = frame::encode(&request, &mut frame).unwrap();
            assert_eq!(&frame[..len], golden);
            assert_eq!(frame::decode::<Request<'_>>(golden), Ok(request));
        }
    }

    /// A launch timeout travels in milliseconds beside the other options.
    #[test]
    fn carries_the_launch_timeout() {
        let request = Request::Configure(Configure {
            options: vec![ConfigureOption::LogLevel(3), ConfigureOption::LaunchTimeout(59_000)],
        });
        let mut frame = [0; MAX_SUBMISSION_WORDS];
        let len = frame::encode(&request, &mut frame).unwrap();
        assert_eq!(frame::decode::<Request<'_>>(&frame[..len]), Ok(request));
    }

    /// An option given twice is refused on both sides: a cluster has one identity.
    #[test]
    fn refuses_repeated_options() {
        let twice = Request::Configure(Configure {
            options: vec![ConfigureOption::Identity(IDENTITY); 2],
        });
        let mut frame = [0; MAX_SUBMISSION_WORDS];
        assert_eq!(
            frame::encode(&twice, &mut frame),
            Err(frame::Error::Encode("an option repeats"))
        );
        let frame = [
            0x163e_25eb_0004_0001,
            0xdfc1_d0bb_0000_0002,
            0xdfc1_d0bb_0302_0109,
            0x0302_0109,
        ];
        assert!(matches!(
            frame::decode::<Request<'_>>(&frame),
            Err(frame::Error::Malformed(_))
        ));
    }
}
