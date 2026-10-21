//! The contract with the signed bootloader. Signed once, it answers one [`Request`], so nothing
//! here may change without a re-signing.
//!
//! - Landing copies and authenticates the same bytes; only the bound digest authorizes entry.
//! - The answer is one [`Completion`](crate::ring::Completion) whose first payload word is a [`Code`].

use crate::ring::header;

/// The request that boots the firmware image staged at `addr..addr + len` under `token`.
pub struct Request {
    pub addr: u64,
    pub len: u64,
    pub token: u64,
}

impl Request {
    pub fn decode([header, addr, len, token]: [u64; 1 + REQUEST_WORDS]) -> Option<Self> {
        let (_, words) = header::unpack(header)?;
        (usize::from(words) == REQUEST_WORDS).then_some(Self { addr, len, token })
    }

    pub const fn encode(&self, id: u32) -> [u64; 1 + REQUEST_WORDS] {
        [header::pack(id, REQUEST_WORDS as u8), self.addr, self.len, self.token]
    }
}

/// The first payload word of the bootloader's answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Code {
    /// The image is in place and about to run.
    Booted = 0,
    /// `addr..addr + len` is outside device memory or the image limit.
    BadRange = 1,
    /// The bytes at `addr` do not open with the token's [`magic`].
    NoImage = 2,
    /// The request had another shape; nothing else is served before the firmware image runs.
    NotBoot = 3,
    /// The image does not match the firmware digest bound into this signed bootloader.
    BadDigest = 4,
}

impl TryFrom<u64> for Code {
    /// A value this side does not name, as it came.
    type Error = u64;

    fn try_from(value: u64) -> Result<Self, u64> {
        match value {
            0 => Ok(Self::Booted),
            1 => Ok(Self::BadRange),
            2 => Ok(Self::NoImage),
            3 => Ok(Self::NotBoot),
            4 => Ok(Self::BadDigest),
            _ => Err(value),
        }
    }
}

/// The first word of a firmware image staged under `token`, so a stale image at a reused address is
/// never booted. An image as built carries token zero.
pub const fn magic(token: u64) -> u64 {
    u64::from_be_bytes(*b"FIRMWARE") ^ token
}

/// Where a firmware image runs in the scratchpad, above the bootloader.
pub const FIRMWARE_BASE: usize = 0x1_0000;
/// First byte reserved for task shared fields by firmware.lds and libpe/tuc.h.
pub const FIRMWARE_END: usize = 0xe_b000;

const REQUEST_WORDS: usize = 3;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Completion;

    /// The signed bootloader's request and answer frames, in the kernel driver's entry layout.
    #[test]
    fn frames_keep_their_shape() {
        let request = [0x0103_0000_0102_0304, 1, 2, 3];
        assert!(Request::decode(request).is_some_and(|decoded| decoded.encode(0x0102_0304) == request));
        assert!(Request::decode([header::pack(1, 2), 1, 2, 3]).is_none());
        let mut replies = Completion::encode(4, &[1, 2, 3, 4, 5]);
        assert_eq!(replies.next(), Some([header::pack(4, 3), 1, 2, 3]));
        let last = replies.next().unwrap();
        assert_eq!((last, replies.next()), ([header::pack(4, 3), 4, 5, 0], None));
        assert!(Completion::decode(last).is_some_and(|reply| reply.id == 4 && reply.payload == [4, 5, 0]));
        assert!(Completion::decode([7, 0, 0, 0]).is_none());
        let codes = [
            Code::Booted,
            Code::BadRange,
            Code::NoImage,
            Code::NotBoot,
            Code::BadDigest,
        ];
        assert!((0..).zip(codes).all(|(value, code)| Code::try_from(value) == Ok(code)));
        assert_eq!(Code::try_from(5), Err(5));
    }
}
