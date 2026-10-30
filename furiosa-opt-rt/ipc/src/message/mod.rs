//! What the host asks a cluster and what the cluster answers. Each request is a type whose
//! [`Call::Reply`] is the only answer it can get.
//!
//! - A request lies on the wire as its call's tag, then the call's fields. The parameters a call
//!   may go without are its options: their count, then each option's tag and payload.
//! - A tag comes from its variant's name alone (`tag::of`), so calls and options come and go
//!   without changing any other frame, and a side refuses a tag it does not know.

use alloc::vec::Vec;
use bincode::{Decode, Encode};

mod answer;
mod configure;
mod profile;
mod request;
mod tag;
pub use answer::{Answer, Failure};
pub use configure::{Configure, ConfigureOption, Identity};
pub use profile::{Profile, Records};
pub use request::MAX_LAUNCH_ARGS;
use tag::tagged;

/// One request the firmware serves, and what it answers on success.
pub trait Call {
    /// The request, as failure messages name it.
    const NAME: &'static str;
    type Reply: Payload;

    fn request(&self) -> Request<'_>;
}

/// Every request, as the firmware decodes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request<'a> {
    Configure(Configure),
    Launch(Launch<'a>),
    Copy(Copy),
    Clock(Clock),
}

tagged! { Request<'a> { Configure, Launch, Copy, Clock } }

/// Runs the executable on the arguments. They come first in the frame, where they sit on a word
/// boundary and the device reads them in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch<'a> {
    /// Argument words, inline or in a device-memory table retained through completion.
    pub args: Args<'a>,
    /// Staged device image and its generation token.
    pub executable: Staged,
    /// What the launch asks for besides running, each option at most once.
    pub options: Vec<LaunchOption>,
}

/// A parameter a launch may go without.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchOption {
    /// What every chunk records, which the reply carries back.
    Profile(Profile),
}

tagged! { LaunchOption { Profile } }

/// Arguments borrowed from this frame or from device memory for one launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Args<'a> {
    Inline(&'a [u64]),
    Indirect { address: u64, count: u32 },
}

/// Device code and its generation identity. `token` distinguishes address reuse.
#[derive(Decode, Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staged {
    /// Device-space byte address of the staged image.
    pub addr: u64,
    /// Length of the staged image in bytes.
    pub len: u64,
    pub token: u64,
}

/// Copies `len` bytes on the cluster's DMA channels between a host range mapped for this cluster,
/// as `furiosa_opt_abi::arch::rngd::dma::Operand` encodes it, and its local memory.
#[derive(Decode, Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Copy {
    pub source: u64,
    pub destination: u64,
    pub len: u64,
}

/// Samples the cluster's ARM counter, in ticks at the platform counter frequency.
#[derive(Decode, Encode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clock;

impl Call for Configure {
    const NAME: &'static str = "configuration";
    type Reply = ();

    fn request(&self) -> Request<'_> {
        Request::Configure(self.clone())
    }
}

impl<'a> Call for Launch<'a> {
    const NAME: &'static str = "launch";
    type Reply = Records;

    fn request(&self) -> Request<'_> {
        Request::Launch(self.clone())
    }
}

impl Call for Copy {
    const NAME: &'static str = "copy";
    type Reply = ();

    fn request(&self) -> Request<'_> {
        Request::Copy(*self)
    }
}

impl Call for Clock {
    const NAME: &'static str = "clock read";
    type Reply = u64;

    fn request(&self) -> Request<'_> {
        Request::Clock(*self)
    }
}

/// What a request answers with on success, as the wire carries it.
pub trait Payload: Encode + Decode<()> {}

impl<T: Encode + Decode<()>> Payload for T {}
