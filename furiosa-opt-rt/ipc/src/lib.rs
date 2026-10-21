//! Messages exchanged between the host runtime and the firmware, and their frame encoding.

#![cfg_attr(not(feature = "std"), no_std)]

use core::time::Duration;

extern crate alloc;
extern crate bincode2 as bincode;
extern crate thiserror_core as thiserror;

pub mod frame;
mod message;
mod status;

pub use frame::{Error, MAX_RESPONSE_WORDS, MAX_SUBMISSION_WORDS};
pub use message::{
    Answer, Args, Call, Clock, Configure, ConfigureOption, Copy, Failure, Identity, Launch, LaunchOption,
    MAX_LAUNCH_ARGS, Payload, Profile, Records, Request, Staged,
};
pub use status::Status;

/// How long a launch may run before the firmware fails it with [`Status::TimedOut`].
pub const LAUNCH_TIMEOUT: Duration = Duration::from_secs(5);
