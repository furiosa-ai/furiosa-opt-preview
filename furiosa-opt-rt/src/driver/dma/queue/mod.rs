//! Host transfer slots behind the driver's ring and the clusters' mailboxes: callers publish their
//! commands and copies, and any poller collects what completed.

mod slot;
mod transfer;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::Waker;

use arrayvec::ArrayVec;

use super::engine::{Copies, Copy, Engine, Flight};
use super::{Command, Transport};
use crate::Result;
use crate::buffer::Waiters;
use crate::driver::Poison;
use crate::driver::mailbox::Mailbox;
use slot::{Slot, Slots, State};
pub(crate) use transfer::Transfer;

/// The first failure rejects every later transfer.
pub(crate) struct DmaQueue {
    slots: Slots,
    /// Submitted slots, in submission order, whose commands the ring or mailboxes had no room for.
    waiting: Mutex<VecDeque<usize>>,
    poison: Arc<Poison>,
    transport: Mutex<Transport>,
    /// The device's clusters, shared with launches; each copies mapped host memory on its own
    /// channels.
    mailboxes: Arc<Mutex<Vec<Mailbox>>>,
    /// The copies the clusters hold. Locked before `mailboxes`.
    flight: Mutex<Flight>,
    /// The device worker, which collects answers no observer polls for.
    worker: Waker,
    /// Woken whenever a transfer ends.
    waiters: Mutex<Waiters>,
}

impl DmaQueue {
    pub(crate) fn open(
        chips: impl IntoIterator<Item = u8>,
        mailboxes: Arc<Mutex<Vec<Mailbox>>>,
        worker: Waker,
    ) -> Result<Self> {
        Ok(Self::new(Transport::open(chips, DEPTH)?, mailboxes, worker))
    }

    pub(crate) fn new(transport: Transport, mailboxes: Arc<Mutex<Vec<Mailbox>>>, worker: Waker) -> Self {
        Self {
            slots: Slots::new(SLOTS),
            waiting: Mutex::default(),
            poison: Arc::default(),
            transport: Mutex::new(transport),
            mailboxes,
            flight: Mutex::default(),
            worker,
            waiters: Mutex::default(),
        }
    }

    /// Collects completed commands, then publishes waiting slots as room frees; true when no
    /// submitted transfer awaits publication or an answer.
    pub(crate) fn complete(&self) -> bool {
        let mut wakers = Vec::new();
        let mut done = |tag, result| wakers.extend(self.finish(tag, result));
        let transport = self.transport.lock().unwrap().complete(&mut done);
        let copies = self.copies().complete(&mut done);
        // An engine that can no longer answer refuses later transfers; what it took still answers.
        if let Err(error) = transport.and(copies) {
            self.poison.set(error);
        }
        wakers.into_iter().for_each(Waker::wake);
        // Waiting slots publish in submission order until one finds no room.
        let mut waiting = self.waiting.lock().unwrap();
        while let Some(&index) = waiting.front() {
            if !self.publish(index) {
                return false;
            }
            waiting.pop_front();
        }
        drop(waiting);
        (self.slots.created()).all(|slot| !matches!(slot.lock().unwrap().state, State::Active { .. }))
    }

    /// Each submitted slot, by index and generation, as [`Self::ended`] takes them.
    pub(crate) fn submitted(&self) -> impl Iterator<Item = (usize, u64)> {
        (self.slots.created().enumerate()).filter_map(|(index, slot)| {
            let slot = slot.lock().unwrap();
            matches!(slot.state, State::Submitted | State::Active { .. }).then_some((index, slot.sequence))
        })
    }

    /// Whether generation `sequence` of slot `index` ended; `waker` is woken at the next end
    /// otherwise.
    pub(crate) fn ended(&self, index: usize, sequence: u64, waker: &Waker) -> bool {
        let ended = || {
            let slot = self.slots[index].lock().unwrap();
            slot.sequence != sequence || !matches!(slot.state, State::Submitted | State::Active { .. })
        };
        ended() || {
            self.waiters.lock().unwrap().register(waker);
            ended()
        }
    }

    /// The first failure, which the driver shares.
    pub(crate) fn poison(&self) -> &Arc<Poison> {
        &self.poison
    }

    /// Hands a slot's commands to the ring, all at once, then its copies to their clusters, as many
    /// as their mailboxes take; false until every one is published.
    fn publish(&self, index: usize) -> bool {
        let mut transport = self.transport.lock().unwrap();
        let mut copies = self.copies();
        let mut slot = self.slots[index].lock().unwrap();
        let first = (slot.sequence << SEQUENCE_SHIFT) | (index as u64 * u64::from(DEPTH));
        if matches!(slot.state, State::Submitted) {
            let commands = (slot.entries.iter().enumerate()).filter_map(|(offset, entry)| match entry {
                Some(Route::Channel(command)) => Some((first + offset as u64, command)),
                _ => None,
            });
            let channels = (slot.entries.iter().enumerate())
                .filter(|(_, entry)| matches!(entry, Some(Route::Channel(_))))
                .fold(0u64, |mask, (offset, _)| mask | 1 << offset);
            let failure = match transport.submit(commands) {
                Ok(taken) if taken < channels.count_ones() as usize => return false,
                Ok(_) => None,
                Err(error) => {
                    let observer = slot.fail(self.poison.set(error).clone());
                    drop(slot);
                    self.waiters.lock().unwrap().take().wake();
                    if let Some(observer) = observer {
                        observer.wake();
                    }
                    return true;
                }
            };
            slot.state = State::Active {
                published: channels,
                received: 0,
                failure,
            };
        }
        let Slot { entries, state, .. } = &mut *slot;
        let State::Active { published, .. } = state else {
            return true;
        };
        let pending = (entries.iter().enumerate())
            .filter_map(|(offset, entry)| match entry {
                Some(Route::Cluster(copy)) if *published & 1 << offset == 0 => Some((offset, copy)),
                _ => None,
            })
            .collect::<ArrayVec<_, { DEPTH as usize }>>();
        let commands = pending.iter().map(|&(offset, copy)| (first + offset as u64, copy));
        let (taken, refused) = match copies.submit(commands) {
            Ok(taken) => (taken, None),
            Err(error) => (0, Some(self.poison.set(error).clone())),
        };
        let offsets = pending
            .iter()
            .map(|&(offset, _)| offset as u64)
            .collect::<ArrayVec<_, { DEPTH as usize }>>();
        drop(pending);
        offsets[..taken].iter().for_each(|offset| *published |= 1 << offset);
        let Some(error) = refused else {
            return taken == offsets.len();
        };
        // Copies no cluster took never run, so they answer now; what other engines took still
        // answers, and the slot ends with the last answer.
        let mut observers = ArrayVec::<Waker, { DEPTH as usize }>::new();
        for &offset in &offsets[taken..] {
            if let State::Active { published, .. } = &mut slot.state {
                *published |= 1 << offset;
            }
            observers.extend(self.receive(&mut slot, offset, Err(error.clone())));
        }
        drop(slot);
        observers.into_iter().for_each(Waker::wake);
        true
    }

    fn copies(&self) -> Copies<'_> {
        Copies {
            flight: self.flight.lock().unwrap(),
            mailboxes: self.mailboxes.lock().unwrap(),
        }
    }
}

/// What moves one entry's bytes.
pub(crate) enum Route {
    /// A kernel driver channel.
    Channel(Command),
    /// A cluster's firmware, on its own channels.
    Cluster(Copy),
}

impl Route {
    fn borrowed(&self) -> bool {
        match self {
            Self::Channel(command) => command.borrowed(),
            Self::Cluster(_) => false,
        }
    }
}

pub(super) const DEPTH: u32 = 64;
/// Transfers the queue tracks at most; slots are created as callers need them, and a caller
/// past this many waits for one to complete.
const SLOTS: usize = 1024;
const SEQUENCE_SHIFT: u32 = DEPTH.ilog2() + SLOTS.ilog2();
const _: () = assert!(DEPTH.is_power_of_two() && SLOTS.is_power_of_two() && DEPTH <= u64::BITS);

/// Transfers on a fake transport, for every module's tests.
#[cfg(test)]
pub(crate) mod fixture {
    use std::task::Waker;

    use super::*;
    use crate::driver::dma::{Direction, Host};
    use crate::{Access, Error};

    pub(crate) fn transport(queue: &DmaQueue) -> crate::driver::dma::fake::Fake {
        let transport = queue.transport.lock().unwrap();
        match &*transport {
            Transport::Fake(fake) => fake.clone(),
            Transport::Driver { .. } => panic!("fake queue required"),
        }
    }

    /// Adds a one-byte read of `buffer` to `transfer`, into `host` when given, else into
    /// borrowed host bytes.
    pub(crate) fn push(transfer: &mut Transfer<'_>, buffer: &crate::Buffer, host: Option<&crate::Pinned>) {
        let host = match host {
            Some(host) => {
                let backing = host.slice(..).pinned.unwrap();
                transfer
                    .access(crate::buffer::Request::host(backing, 1, Access::Write))
                    .unwrap();
                Host::Owned(Arc::clone(backing.dmabuf), 0)
            }
            None => Host::Borrowed(0),
        };
        let command = Command::new(0, Direction::Read, host, 0, 1);
        transfer.access((buffer.slice(..).unwrap(), Access::Read)).unwrap();
        transfer.push(Route::Channel(command));
    }

    /// A command's segments, each as its `(src, dst, len)`.
    pub(crate) type Segments = Vec<(u64, u64, u64)>;

    /// Each driver command the queue's slots hold: its chip and its segments.
    pub(crate) fn commands(queue: &DmaQueue) -> Vec<(u8, Segments)> {
        (queue.slots.created())
            .flat_map(|slot| {
                let slot = slot.lock().unwrap();
                (slot.entries.iter())
                    .filter_map(|entry| match entry {
                        Some(Route::Channel(command)) => Some((
                            command.chip(),
                            (command.segments().iter())
                                .map(|segment| (segment.src, segment.dst, segment.len))
                                .collect(),
                        )),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The tag of `transfer`'s first command.
    pub(crate) fn tag(transfer: &Transfer<'_>) -> u64 {
        (transfer.sequence << SEQUENCE_SHIFT) | (transfer.index as u64 * u64::from(DEPTH))
    }

    /// The engine answers command `tag`; the observer to wake once that ends its transfer.
    pub(crate) fn finish(queue: &DmaQueue, tag: u64, result: Result<()>) -> Option<Waker> {
        queue.finish(tag, result)
    }

    /// `tag`'s transfer as if its engine took every command, with `failure` already recorded.
    pub(crate) fn activate(queue: &DmaQueue, tag: u64, failure: Option<Error>) {
        let index = (tag / u64::from(DEPTH) % SLOTS as u64) as usize;
        queue.slots[index].lock().unwrap().state = State::Active {
            published: u64::MAX,
            received: 0,
            failure,
        };
    }
}
