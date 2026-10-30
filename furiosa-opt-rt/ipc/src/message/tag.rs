//! Tags, which name a call or an option on the wire, and the option lists built from them. A tag
//! comes from its variant's name alone, so variants come and go without changing other frames.

use alloc::vec::Vec;
use core::mem::discriminant;

use bincode::de::BorrowDecoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};

/// The tag of the variant `name`, its 32-bit FNV-1a hash. A name an enum drops is never given to a
/// new variant: a peer still sending it must be refused, not read as the new one.
pub(super) const fn of(name: &str) -> u32 {
    const BASIS: u32 = 0x811c_9dc5;
    const PRIME: u32 = 0x0100_0193;
    let (bytes, mut hash, mut index) = (name.as_bytes(), BASIS, 0);
    while index < bytes.len() {
        hash = (hash ^ bytes[index] as u32).wrapping_mul(PRIME);
        index += 1;
    }
    hash
}

/// Encodes the enum `$name`, each of whose variants holds one payload, as the variant's tag, then
/// the payload; decoding refuses an unknown tag. Compiling fails if two variants share a tag.
macro_rules! tagged {
    ($name:ident $(<$lifetime:lifetime>)? { $($variant:ident),+ $(,)? }) => {
        impl $(<$lifetime>)? $name $(<$lifetime>)? {
            /// Every variant's tag, in declaration order.
            pub(crate) const TAGS: &'static [u32] = &[$($crate::message::tag::of(stringify!($variant))),+];
        }

        const _: () = assert!(
            $crate::message::tag::distinct($name::TAGS),
            concat!("two ", stringify!($name), " variants share a tag"),
        );

        impl $(<$lifetime>)? bincode::Encode for $name $(<$lifetime>)? {
            fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
                match self {
                    $(Self::$variant(payload) => {
                        let tag = const { $crate::message::tag::of(stringify!($variant)) };
                        bincode::Encode::encode(&(tag, payload), encoder)
                    })+
                }
            }
        }

        impl<$($lifetime,)? 'de $(: $lifetime)?, Context> bincode::BorrowDecode<'de, Context>
            for $name $(<$lifetime>)?
        {
            fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
                decoder: &mut D,
            ) -> Result<Self, bincode::error::DecodeError> {
                let tag = <u32 as bincode::Decode<Context>>::decode(decoder)?;
                $(if tag == const { $crate::message::tag::of(stringify!($variant)) } {
                    return Ok(Self::$variant(bincode::BorrowDecode::borrow_decode(decoder)?));
                })+
                Err(bincode::error::DecodeError::Other(concat!("unknown ", stringify!($name))))
            }
        }
    };
}
pub(super) use tagged;

/// Writes `options` as a call's options: their count, then each option's tag and payload.
pub(super) fn encode_options<T: Encode, E: Encoder>(options: &[T], encoder: &mut E) -> Result<(), EncodeError> {
    if repeats(options) {
        return Err(EncodeError::Other("an option repeats"));
    }
    // A count past `u32` never reaches the wire: the options it counts overflow the frame first.
    (options.len() as u32).encode(encoder)?;
    options.iter().try_for_each(|option| option.encode(encoder))
}

/// Reads the options [`encode_options`] writes, refusing one this side does not know or one given
/// twice.
pub(super) fn decode_options<'de, T, D>(decoder: &mut D) -> Result<Vec<T>, DecodeError>
where
    D: BorrowDecoder<'de>,
    T: BorrowDecode<'de, D::Context>,
{
    let count = u32::decode(decoder)?;
    let options = (0..count)
        .map(|_| T::borrow_decode(decoder))
        .collect::<Result<Vec<_>, _>>()?;
    if repeats(&options) {
        return Err(DecodeError::Other("an option repeats"));
    }
    Ok(options)
}

/// Whether no two of `tags` are equal.
pub(super) const fn distinct(tags: &[u32]) -> bool {
    let mut index = 0;
    while index < tags.len() {
        let mut other = index + 1;
        while other < tags.len() {
            if tags[index] == tags[other] {
                return false;
            }
            other += 1;
        }
        index += 1;
    }
    true
}

/// Whether two of `options` are the same option, whatever their payloads.
fn repeats<T>(options: &[T]) -> bool {
    (options.iter().enumerate()).any(|(index, option)| {
        options[..index]
            .iter()
            .any(|other| discriminant(other) == discriminant(option))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Args, ConfigureOption, LaunchOption, Request};

    /// Every enum's tags differ, and the check that proves it when compiling catches a repeat.
    #[test]
    fn distinguishes_the_variants_of_each_enum() {
        for tags in [Request::TAGS, LaunchOption::TAGS, ConfigureOption::TAGS, Args::TAGS] {
            assert!(distinct(tags));
        }
        assert!(!distinct(&[of("Launch"), of("Copy"), of("Launch")]));
    }
}
