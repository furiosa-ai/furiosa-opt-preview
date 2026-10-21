//! Landing a staged firmware image in the scratchpad under its signed digest.

#![no_std]

use furiosa_opt_abi::arch::rngd::DEVICE_MEMORY;
use furiosa_opt_abi::bootloader::{Code, FIRMWARE_BASE, FIRMWARE_END, Request, magic};
use sha2::{Digest, Sha256};

/// A staged firmware image that fits: aligned words inside device memory and the image limit, at
/// least its magic and one more, staged under `token`.
#[derive(Debug, PartialEq, Eq)]
pub struct Image {
    words: usize,
    token: u64,
}

impl TryFrom<Request> for Image {
    type Error = Code;

    /// Refuses with [`Code::BadRange`] a request whose range cannot hold an image.
    fn try_from(Request { addr, len, token }: Request) -> Result<Self, Code> {
        let fits = len >= 16
            && len <= IMAGE_LIMIT as u64
            && len.is_multiple_of(8)
            && addr.is_multiple_of(8)
            && addr >= DEVICE_MEMORY.start
            && addr <= DEVICE_MEMORY.end - len;
        fits.then_some(Self {
            words: len as usize / 8,
            token,
        })
        .ok_or(Code::BadRange)
    }
}

impl Image {
    /// Copies and authenticates each source word once against `digest`; only [`Code::Booted`]
    /// authorizes entry. A failed digest may leave destination bytes and returns [`Code::BadDigest`].
    pub fn land(
        self,
        digest: &[u8; 32],
        mut read: impl FnMut(usize) -> u64,
        mut write: impl FnMut(usize, u64),
    ) -> Code {
        if read(0) != magic(self.token) {
            return Code::NoImage;
        }
        let mut hash = Sha256::new();
        for word in 0..self.words {
            // The image as built opens with token zero's magic; the staged copy carries its token.
            let image = if word == 0 { magic(0) } else { read(word) };
            write(word, image);
            hash.update(image.to_le_bytes());
        }
        if hash.finalize().as_slice() == digest {
            Code::Booted
        } else {
            Code::BadDigest
        }
    }
}

/// The most bytes a firmware image may span.
const IMAGE_LIMIT: usize = FIRMWARE_END - FIRMWARE_BASE;

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: u64 = 0x1234_5678_9abc_def0;
    const IMAGE: [u64; 3] = [magic(0), 0x0102_0304_0506_0708, 0x1112_1314_1516_1718];

    /// Only the bound digest boots, over the same bytes it copies, each read once.
    #[test]
    fn lands_only_the_bound_image() {
        let mut staged = IMAGE;
        staged[0] ^= TOKEN;
        staged[1] ^= 1;
        let mut destination = [0; 3];

        let digest: [u8; 32] = Sha256::digest(IMAGE.map(u64::to_le_bytes).as_flattened()).into();
        let image = || {
            Image::try_from(Request {
                addr: DEVICE_MEMORY.start,
                len: 8 * IMAGE.len() as u64,
                token: TOKEN,
            })
            .unwrap()
        };
        let code = image().land(&digest, |word| staged[word], |word, value| destination[word] = value);

        assert_eq!(code, Code::BadDigest);

        staged[1] ^= 1;
        let mut reads = [0; 3];
        let code = image().land(
            &digest,
            |word| {
                reads[word] += 1;
                staged[word]
            },
            |word, value| destination[word] = value,
        );

        assert_eq!(code, Code::Booted);
        assert_eq!(reads, [1; 3]);
        assert_eq!(destination, IMAGE);
    }

    /// A boot request names only an aligned image inside device memory and the image limit.
    #[test]
    fn accepts_only_device_memory_that_fits() {
        let (start, end) = (DEVICE_MEMORY.start, DEVICE_MEMORY.end);
        let image = |addr, len| Image::try_from(Request { addr, len, token: 0 });
        assert!(image(start, 16).is_ok());
        assert!(image(end - 16, 16).is_ok());
        assert_eq!(image(end - 8, 16), Err(Code::BadRange), "past the end");
        assert_eq!(image(start - 8, 16), Err(Code::BadRange), "below the start");
        assert_eq!(image(start + 4, 16), Err(Code::BadRange), "unaligned");
        assert_eq!(
            image(start, IMAGE_LIMIT as u64 + 8),
            Err(Code::BadRange),
            "over the limit"
        );
    }
}
