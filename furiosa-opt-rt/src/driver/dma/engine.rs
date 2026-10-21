//! The two ways host transfers move bytes, behind one contract: tagged commands go in, and each
//! comes back answered under its tag.

use std::sync::{Arc, MutexGuard};

use super::Direction;
use crate::driver::kernel::Segment;
use crate::driver::mailbox::{self, Mailbox, Sent};
use crate::pinned::{PinnedLease, Registration};
use crate::{Error, Result};

/// One way bytes move: commands go in under a caller's tag, and each one's result comes back
/// under that tag once it finishes, in any order.
pub(crate) trait Engine {
    type Command;

    /// Submits `commands` in order and returns how many it took; it stops at the first it has no
    /// room for. An error means none of them will answer.
    fn submit<'a>(&mut self, commands: impl Iterator<Item = (u64, &'a Self::Command)>) -> Result<usize>
    where
        Self::Command: 'a;

    /// Hands `done` the result of every command that finished since the last call, each command's
    /// exactly once.
    fn complete(&mut self, done: impl FnMut(u64, Result<()>));
}

/// A copy cluster `cluster`'s firmware runs on its own PDMA channels, between `len` device bytes
/// at `device` and the host bytes `host` leases. The firmware reaches the host bytes through the
/// registration's table, which it walks until the copy answers.
pub(crate) struct Copy {
    cluster: usize,
    direction: Direction,
    registration: Arc<Registration>,
    host: PinnedLease,
    /// Where the cluster sees its chip's memory, which the registration's table lies in.
    window: u64,
    device: u64,
    len: u64,
}

impl Copy {
    pub(crate) fn new(
        cluster: usize,
        direction: Direction,
        registration: Arc<Registration>,
        host: PinnedLease,
        window: u64,
        device: u64,
        len: u64,
    ) -> Self {
        Self {
            cluster,
            direction,
            registration,
            host,
            window,
            device,
            len,
        }
    }

    /// The request that has the cluster run this copy.
    fn request(&self) -> furiosa_opt_ipc::Copy {
        let offset = u32::try_from(self.host.offset()).expect("a mapped allocation spans at most u32::MAX bytes");
        let host = self.registration.operand(self.window, self.cluster, offset);
        let Segment { src, dst, len } = self.direction.segment(host, self.device, self.len);
        furiosa_opt_ipc::Copy {
            source: src,
            destination: dst,
            len,
        }
    }
}

/// The copies the clusters hold, by the request id their mailbox answers under.
#[derive(Default)]
pub(crate) struct Flight {
    sent: Vec<(u64, usize, Sent<()>)>,
    /// Copies a mailbox refused: never in flight, answered with the refusal.
    refused: Vec<(u64, Error)>,
}

/// [`Copy`] commands on the clusters' mailboxes.
pub(crate) struct Copies<'a> {
    pub(crate) flight: MutexGuard<'a, Flight>,
    pub(crate) mailboxes: MutexGuard<'a, Vec<Mailbox>>,
}

impl Engine for Copies<'_> {
    type Command = Copy;

    fn submit<'a>(&mut self, commands: impl Iterator<Item = (u64, &'a Copy)>) -> Result<usize> {
        let mut taken = 0;
        for (tag, copy) in commands {
            match self.mailboxes[copy.cluster].prepare(&copy.request()) {
                Ok(Some(prepared)) => self.flight.sent.push((tag, copy.cluster, prepared.send())),
                Ok(None) => break,
                Err(error) => self.flight.refused.push((tag, error)),
            }
            taken += 1;
        }
        Ok(taken)
    }

    fn complete(&mut self, mut done: impl FnMut(u64, Result<()>)) {
        let Flight { sent, refused } = &mut *self.flight;
        refused.drain(..).for_each(|(tag, error)| done(tag, Err(error)));
        let mailboxes = &mut *self.mailboxes;
        sent.retain(|&(tag, cluster, sent)| {
            let result = match mailboxes[cluster].receive(sent) {
                Ok(None) => return true,
                Ok(Some(answer)) => {
                    answer.map_err(|failure| mailbox::refused::<furiosa_opt_ipc::Copy>(cluster, &failure))
                }
                Err(error) => Err(error),
            };
            done(tag, result);
            false
        });
    }
}
