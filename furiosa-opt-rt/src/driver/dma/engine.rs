//! The two ways host transfers move bytes, behind one contract: tagged commands go in, and each
//! comes back answered under its tag.

use std::sync::{Arc, MutexGuard};

use super::Direction;
use crate::driver::kernel::Segment;
use crate::driver::mailbox::{self, Mailbox, Sent};
use crate::pinned::Registration;
use crate::{Error, Result};

/// One way bytes move: commands go in under a caller's tag, and each one's result comes back
/// under that tag once it finishes, in any order.
pub(crate) trait Engine {
    type Command;

    /// Submits `commands` in order and returns how many it took, each of which answers once; it
    /// stops at the first it has no room for. An error means it took none.
    fn submit<'a>(&mut self, commands: impl Iterator<Item = (u64, &'a Self::Command)>) -> Result<usize>
    where
        Self::Command: 'a;

    /// Hands `done` the result of every command that finished since the last call, each command's
    /// exactly once. An error means the engine can no longer be relied on to answer what it took;
    /// the commands it took still answer through `done` if they can.
    fn complete(&mut self, done: impl FnMut(u64, Result<()>)) -> Result<()>;
}

/// A firmware copy retaining the registration table it walks until completion.
pub(crate) struct Copy {
    cluster: usize,
    direction: Direction,
    registration: Arc<Registration>,
    host: usize,
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
        host: usize,
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
        let offset = u32::try_from(self.host).expect("a mapped allocation spans at most u32::MAX bytes");
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
        let mut commands = commands.peekable();
        if commands.peek().is_some() && self.mailboxes.is_empty() {
            return Err(stopped());
        }
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

    fn complete(&mut self, mut done: impl FnMut(u64, Result<()>)) -> Result<()> {
        let Flight { sent, refused } = &mut *self.flight;
        refused.drain(..).for_each(|(tag, error)| done(tag, Err(error)));
        let mailboxes = &mut *self.mailboxes;
        let mut failure = None;
        sent.retain(|&(tag, cluster, sent)| {
            if failure.is_some() {
                return true;
            }
            if mailboxes.is_empty() {
                failure = Some(stopped());
                return true;
            }
            let result = match mailboxes[cluster].receive(sent) {
                Ok(None) => return true,
                Ok(Some(answer)) => {
                    answer.map_err(|failure| mailbox::refused::<furiosa_opt_ipc::Copy>(cluster, &failure))
                }
                Err(error) => {
                    failure = Some(error);
                    return true;
                }
            };
            done(tag, result);
            false
        });
        if let Some(error) = failure {
            // Closing the reservations stops and drains firmware DMA before copies release memory.
            mailboxes.clear();
            sent.drain(..).for_each(|(tag, _, _)| done(tag, Err(error.clone())));
            return Err(error);
        }
        Ok(())
    }
}

/// Why nothing reaches the firmware once a mailbox failure closed every cluster's mailbox.
pub(crate) fn stopped() -> Error {
    Error::Device("DMA mailboxes have stopped".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A broken mailbox stops its DMA before releasing any copy's host memory.
    #[test]
    fn stops_copies_on_broken_mailbox() {
        let device = crate::test::device(&[0], 8);
        let host = crate::test::mapped(&device, vec![0], 4096);
        let backing = host.slice(..).pinned.unwrap();
        let copy = Copy::new(
            0,
            Direction::Read,
            Arc::clone(backing.registration.unwrap()),
            0,
            0,
            0,
            4096,
        );
        let (mailbox, file) = crate::test::mailbox();
        let mailboxes = Mutex::new(vec![mailbox]);
        let flight = Mutex::new(Flight::default());
        let mut copies = Copies {
            flight: flight.lock().unwrap(),
            mailboxes: mailboxes.lock().unwrap(),
        };
        assert_eq!(copies.submit(std::iter::once((1, &copy))).unwrap(), 1);
        crate::test::respond(&file, &[(999, furiosa_opt_ipc::Answer(Ok(())))]);
        let mut results = Vec::new();
        assert!(copies.complete(|tag, result| results.push((tag, result))).is_err());
        assert!(copies.mailboxes.is_empty());
        assert!(matches!(results.as_slice(), [(1, Err(Error::Device(_)))]));
        assert!(copies.submit(std::iter::once((2, &copy))).is_err());
    }
}
