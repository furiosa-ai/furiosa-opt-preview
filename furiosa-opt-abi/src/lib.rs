//! Device executable and hardware contracts.

#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "bincode")]
extern crate bincode2 as bincode;
extern crate thiserror_core as thiserror;

use core::future::Future;
use core::task::{Context, Poll, Waker};

pub mod bootloader;
pub mod doorbell;
pub mod executable;
pub mod memory;
pub mod mmio;
pub mod ring;

/// Chip-specific layouts and limits.
pub mod arch {
    pub mod rngd;
}

/// Polls `future` until it is ready, spinning between polls: the hardware units answer `Poll` and
/// wake no one, so a caller with nothing else to do waits like this.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        core::hint::spin_loop();
    }
}

/// Declares an enum of numbered values and its `TryFrom` the other side decodes with, from one
/// list, so a value the enum names cannot go undecoded. Go-to-definition on a variant lands in
/// the invocation, which still reads as a plain enum.
#[macro_export]
macro_rules! enum_try_from {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident: $repr:ident {
            $($(#[$variant_meta:meta])* $variant:ident = $value:literal,)*
        }
    ) => {
        $(#[$meta])*
        #[repr($repr)]
        $vis enum $name {
            $($(#[$variant_meta])* $variant = $value,)*
        }

        impl TryFrom<$repr> for $name {
            /// A value this side does not name, as it came.
            type Error = $repr;

            fn try_from(value: $repr) -> Result<Self, $repr> {
                match value {
                    $($value => Ok(Self::$variant),)*
                    _ => Err(value),
                }
            }
        }
    };
}

/// Number of chips and processing elements per chip in a compiled program or opened device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Topology {
    pub chips: u8,
    pub pes: u8,
}

impl From<(u8, u8)> for Topology {
    fn from((chips, pes): (u8, u8)) -> Self {
        Self { chips, pes }
    }
}
