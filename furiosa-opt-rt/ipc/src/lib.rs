//! Messages exchanged between the host runtime and the firmware, and their frame encoding.

#![cfg_attr(not(feature = "std"), no_std)]

use core::time::Duration;

extern crate alloc;
extern crate bincode2 as bincode;
extern crate thiserror_core as thiserror;

mod message;
mod status;
pub mod wire;

pub use message::{Args, Request, Response, Staged};
pub use status::Status;
pub use wire::{Error, MAX_RESPONSE_WORDS, MAX_SUBMISSION_WORDS};

/// How long a launch may run before the firmware fails it with [`Status::TimedOut`].
pub const LAUNCH_TIMEOUT: Duration = Duration::from_secs(5);
