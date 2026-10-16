//! Landing a staged firmware image in the scratchpad under its signed digest.

#![no_std]

use furiosa_opt_abi::arch::rngd::DEVICE_MEMORY;
use furiosa_opt_abi::bootloader::{Code, FIRMWARE_BASE, FIRMWARE_END, magic};
use sha2::{Digest, Sha256};

/// The most bytes a firmware image may span.
const IMAGE_LIMIT: usize = FIRMWARE_END - FIRMWARE_BASE;

/// Whether `addr..addr + len` can hold a firmware image: inside device memory and the limit.
pub const fn in_range(addr: u64, len: u64) -> bool {
    len >= 16
        && len <= IMAGE_LIMIT as u64
        && len.is_multiple_of(8)
        && addr.is_multiple_of(8)
        && addr >= DEVICE_MEMORY.start
        && addr <= DEVICE_MEMORY.end - len
}

/// A staged firmware image and the digest that authorizes it.
pub struct Image<'a> {
    pub words: usize,
    pub token: u64,
    pub digest: &'a [u8; 32],
}

impl Image<'_> {
    /// Copies and authenticates each source word once; only [`Code::Booted`] authorizes entry.
    /// A failed digest may leave destination bytes and returns [`Code::BadDigest`].
    pub fn land(self, mut read: impl FnMut(usize) -> u64, mut write: impl FnMut(usize, u64)) -> Code {
        if self.words < 2 {
            return Code::BadRange;
        }
        let first = read(0);
        if first != magic(self.token) {
            return Code::NoImage;
        }

        let mut digest = Sha256::new();
        for word in 0..self.words {
            let image = if word == 0 { magic(0) } else { read(word) };
            write(word, image);
            digest.update(image.to_le_bytes());
        }
        if digest.finalize().as_slice() == self.digest {
            Code::Booted
        } else {
            Code::BadDigest
        }
    }
}

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
        let image = || Image {
            words: IMAGE.len(),
            token: TOKEN,
            digest: &digest,
        };
        let code = image().land(|word| staged[word], |word, value| destination[word] = value);

        assert_eq!(code, Code::BadDigest);

        staged[1] ^= 1;
        let mut reads = [0; 3];
        let code = image().land(
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
        assert!(in_range(start, 16));
        assert!(in_range(end - 16, 16));
        assert!(!in_range(end - 8, 16), "past the end");
        assert!(!in_range(start - 8, 16), "below the start");
        assert!(!in_range(start + 4, 16), "unaligned");
        assert!(!in_range(start, IMAGE_LIMIT as u64 + 8), "over the limit");
    }
}
