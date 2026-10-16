//! Host DMA, issued through queue slots: driver commands over io_uring, and copies of
//! mapped host memory the clusters run on their own channels.

mod command;
mod queue;
mod transport;
mod uring;
use command::Channel;
pub(crate) use command::{Command, Direction, Host};
pub(crate) use queue::{DmaQueue, Route, Transfer};
pub(crate) use transport::Transport;
#[cfg(test)]
pub(crate) use transport::fake;
use uring::Ring;
