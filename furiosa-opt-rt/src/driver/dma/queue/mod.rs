//! Host transfer slots behind the driver's ring and the clusters' links: callers publish their
//! commands and copies, and any poller retires answers.

mod slot;
mod transfer;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::task::Waker;

use arrayvec::ArrayVec;
use furiosa_opt_ipc::{Request, Response};

use super::{Command, Transport};
use crate::arch::rngd::Segment;
use crate::buffer::Waiters;
use crate::driver::link::Link;
use crate::pinned::{HostLease, Mapping};
use crate::{Error, Result};
use slot::{Slot, State};
pub(crate) use transfer::Transfer;

pub(super) const DEPTH: u32 = 64;
/// Transfers the queue tracks at most; slots are created as callers need them, and a caller
/// past this many waits for one to retire.
const SLOTS: usize = 1024;
const SEQUENCE_SHIFT: u32 = DEPTH.ilog2() + SLOTS.ilog2();
const _: () = assert!(DEPTH.is_power_of_two() && SLOTS.is_power_of_two() && DEPTH <= u64::BITS);

/// The first failure rejects every later transfer.
pub(crate) struct DmaQueue {
    slots: Box<[OnceLock<Mutex<Slot>>]>,
    /// Submitted slots, in submission order, whose commands the ring or links had no room for.
    waiting: Mutex<VecDeque<usize>>,
    pub(crate) failure: OnceLock<Error>,
    transport: Mutex<Transport>,
    /// The device's clusters, shared with launches; each copies mapped host memory on its own
    /// channels.
    links: Arc<Mutex<Vec<Link>>>,
    /// The device worker, which retires answers no observer polls for.
    worker: Waker,
    /// Woken whenever a transfer ends.
    waiters: Mutex<Waiters>,
}

/// What moves one entry's bytes.
pub(crate) enum Route {
    /// A kernel driver channel, answering on the ring.
    Channel(Command),
    /// Cluster `cluster`'s firmware, on its own channels, answering on its link under `id` once
    /// published. The host side is an operand of the mapping, whose table the firmware walks.
    Cluster {
        cluster: usize,
        segment: Segment,
        _mapping: Arc<Mapping>,
        _host: HostLease,
        id: Option<u32>,
    },
}

impl DmaQueue {
    pub(crate) fn open(
        chips: impl IntoIterator<Item = u8>,
        links: Arc<Mutex<Vec<Link>>>,
        worker: Waker,
    ) -> Result<Self> {
        Ok(Self::new(Transport::open(chips, DEPTH)?, links, worker))
    }

    pub(crate) fn new(transport: Transport, links: Arc<Mutex<Vec<Link>>>, worker: Waker) -> Self {
        Self {
            slots: (0..SLOTS).map(|_| OnceLock::new()).collect(),
            waiting: Mutex::default(),
            failure: OnceLock::new(),
            transport: Mutex::new(transport),
            links,
            worker,
            waiters: Mutex::default(),
        }
    }

    /// Each submitted transfer's slot and generation, as [`Self::ended`] takes them.
    pub(crate) fn submitted(&self) -> Vec<(usize, u64)> {
        (self.created().enumerate())
            .filter_map(|(index, slot)| {
                let slot = slot.lock().unwrap();
                matches!(slot.state, State::Submitted | State::Active { .. }).then_some((index, slot.sequence))
            })
            .collect()
    }

    /// Whether every transfer of `submitted` ended; `waker` is woken at the next end otherwise.
    pub(crate) fn ended(&self, submitted: &[(usize, u64)], waker: &Waker) -> bool {
        let ended = || {
            submitted.iter().all(|&(index, sequence)| {
                let slot = self.slot(index).lock().unwrap();
                slot.sequence != sequence || !matches!(slot.state, State::Submitted | State::Active { .. })
            })
        };
        ended() || {
            self.waiters.lock().unwrap().register(waker);
            ended()
        }
    }

    /// Retires answered commands, then publishes waiting slots as room frees; true when no
    /// submitted transfer awaits publication or an answer.
    pub(crate) fn progress(&self) -> bool {
        let mut wakers = Vec::new();
        self.transport.lock().unwrap().retire(|tag, result| {
            let result = result
                .map(drop)
                .map_err(|error| Error::Device(format!("DMA command failed: {error}")));
            wakers.extend(self.complete(tag, result));
        });
        wakers.into_iter().for_each(Waker::wake);
        let mut wakers = Vec::new();
        let mut links = self.links.lock().unwrap();
        for index in 0..self.created().count() {
            for (tag, result) in self.answers(&mut links, index) {
                wakers.extend(self.complete(tag, result));
            }
        }
        drop(links);
        wakers.into_iter().for_each(Waker::wake);
        self.publish_waiting()
            && self
                .created()
                .all(|slot| !matches!(slot.lock().unwrap().state, State::Active { .. }))
    }

    /// The slots created so far; a slot is created only after every earlier one.
    fn created(&self) -> impl Iterator<Item = &Mutex<Slot>> {
        self.slots.iter().map_while(OnceLock::get)
    }

    fn slot(&self, index: usize) -> &Mutex<Slot> {
        self.slots[index].get().expect("a slot is created before it is used")
    }

    /// Publishes waiting slots in submission order until one finds no room; true when none
    /// waits.
    fn publish_waiting(&self) -> bool {
        let mut waiting = self.waiting.lock().unwrap();
        while let Some(&index) = waiting.front() {
            if !self.publish(index) {
                return false;
            }
            waiting.pop_front();
        }
        true
    }

    /// The answers slot `index`'s published copies have, by tag.
    fn answers(&self, links: &mut [Link], index: usize) -> ArrayVec<(u64, Result<()>), { DEPTH as usize }> {
        let slot = self.slot(index).lock().unwrap();
        let State::Active { received, .. } = slot.state else {
            return ArrayVec::new();
        };
        let first = (slot.sequence << SEQUENCE_SHIFT) | (index as u64 * u64::from(DEPTH));
        let pending = slot
            .entries
            .iter()
            .enumerate()
            .filter_map(|(offset, entry)| match entry {
                Route::Cluster {
                    cluster, id: Some(id), ..
                } if received & 1 << offset == 0 => Some((first + offset as u64, *cluster, *id)),
                _ => None,
            });
        pending
            .filter_map(|(tag, cluster, id)| {
                let result = match links[cluster].poll(id).transpose()? {
                    Ok(Response::Copy) => Ok(()),
                    Ok(Response::Error { status, message }) => {
                        Err(crate::driver::link::failure(cluster, "copy", status, &message))
                    }
                    Ok(other) => Err(Error::Device(format!("rank {cluster} answered a copy with {other:?}"))),
                    Err(error) => Err(error),
                };
                Some((tag, result))
            })
            .collect()
    }

    /// Hands a slot's commands to the ring, all at once, then its copies to their clusters, as many
    /// as their links take; false until every one is published.
    fn publish(&self, index: usize) -> bool {
        let mut transport = self.transport.lock().unwrap();
        let mut links = self.links.lock().unwrap();
        let mut slot = self.slot(index).lock().unwrap();
        let first = (slot.sequence << SEQUENCE_SHIFT) | (index as u64 * u64::from(DEPTH));
        if matches!(slot.state, State::Submitted) {
            let commands = slot
                .entries
                .iter()
                .enumerate()
                .filter_map(|(offset, entry)| match entry {
                    Route::Channel(command) => Some((first + offset as u64, command)),
                    Route::Cluster { .. } => None,
                });
            let failure = match transport.submit(commands) {
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return false,
                result => result
                    .err()
                    .map(|error| Error::Device(format!("the DMA ring is broken: {error}"))),
            };
            if let Some(error) = &failure {
                self.failure.get_or_init(|| error.clone());
            }
            slot.state = State::Active { received: 0, failure };
        }
        // A copy its link refuses is never in flight, so it is answered here, with the refusal.
        let (mut room, mut refused) = (true, ArrayVec::<_, { DEPTH as usize }>::new());
        let Slot { entries, state, .. } = &mut *slot;
        let answered = match state {
            State::Active { received, .. } => *received,
            _ => u64::MAX,
        };
        for (offset, entry) in entries.iter_mut().enumerate() {
            let Route::Cluster {
                cluster,
                segment,
                id: id @ None,
                ..
            } = entry
            else {
                continue;
            };
            if answered & 1 << offset != 0 {
                continue;
            }
            let copy = Request::Copy {
                source: segment.src,
                destination: segment.dst,
                len: segment.len,
            };
            match links[*cluster].prepare(&copy) {
                Ok(Some(prepared)) => *id = Some(prepared.publish()),
                Ok(None) => {
                    room = false;
                    break;
                }
                Err(error) => refused.push((offset as u64, error)),
            }
        }
        let woken = (refused.into_iter())
            .filter_map(|(offset, error)| {
                self.failure.get_or_init(|| error.clone());
                self.receive(&mut slot, offset, Err(error))
            })
            .collect::<ArrayVec<_, { DEPTH as usize }>>();
        drop((slot, links, transport));
        woken.into_iter().for_each(Waker::wake);
        room
    }
}

impl Route {
    fn borrowed(&self) -> bool {
        match self {
            Self::Channel(command) => command.borrowed(),
            Self::Cluster { .. } => false,
        }
    }
}

#[cfg(test)]
mod fixture {
    use super::*;
    use crate::Access;
    use crate::driver::dma::{Direction, Host};

    pub(super) fn push(transfer: &mut Transfer<'_>, buffer: &crate::Buffer, host: Option<&crate::Pinned>) {
        let host = match host {
            Some(host) => Host::Owned(host.slice(..).backing.unwrap().retain(), 0),
            None => Host::Borrowed(0),
        };
        let command = Command::new(0, Direction::Read, host, 0, 1);
        transfer.access(buffer.slice(..).unwrap(), Access::Read).unwrap();
        transfer.push(Route::Channel(command));
    }

    pub(super) fn tag(transfer: &Transfer<'_>) -> u64 {
        (transfer.sequence << SEQUENCE_SHIFT) | (transfer.index as u64 * u64::from(DEPTH))
    }
}
