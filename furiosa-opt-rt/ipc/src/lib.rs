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

/// The kernel driver's IPC deadline by default (`pe_ipc_timeout_ms`), assumed where it cannot be
/// read.
pub const DRIVER_DEADLINE: Duration = Duration::from_secs(60);

/// How long a launch may run under the driver's IPC `deadline` before the firmware fails it with
/// [`Status::TimedOut`]: one second short of the deadline, so the answer naming the timeout reaches
/// the host before the driver gives up on the cluster. A shorter limit would fail long launches the
/// driver still allows.
pub const fn launch_timeout(deadline: Duration) -> Duration {
    deadline.saturating_sub(Duration::from_secs(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A launch gets a second less than the driver's deadline, and none from a deadline of a second.
    #[test]
    fn times_launches_out_before_the_driver() {
        assert_eq!(launch_timeout(DRIVER_DEADLINE), Duration::from_secs(59));
        assert!(launch_timeout(Duration::from_secs(1)).is_zero());
    }
}
