use std::future::poll_fn;
use std::ops::Index;
use std::sync::{Mutex, OnceLock};
use std::task::{Poll, Waker};

use arrayvec::ArrayVec;

use super::{DEPTH, DmaQueue, Route, SEQUENCE_SHIFT, SLOTS, Transfer};
use crate::buffer::Lease;
use crate::{Error, Result};

impl DmaQueue {
    pub(crate) async fn acquire(&self) -> Result<Transfer<'_>> {
        poll_fn(|context| {
            if let Some(error) = self.poison.get() {
                return Poll::Ready(Err(error.clone()));
            }
            for (index, slot) in self.slots.each().enumerate() {
                let mut slot = slot.lock().unwrap();
                if matches!(slot.state, State::Idle | State::Complete(Ok(()))) {
                    slot.sequence = slot
                        .sequence
                        .checked_add(1)
                        .filter(|sequence| *sequence <= u64::MAX >> SEQUENCE_SHIFT)
                        .ok_or_else(|| Error::Device("DMA generation exhausted".into()))?;
                    slot.state = State::Preparing;
                    return Poll::Ready(Ok(Transfer {
                        queue: self,
                        index,
                        sequence: slot.sequence,
                    }));
                }
            }
            self.worker.wake_by_ref();
            context.waker().wake_by_ref();
            Poll::Pending
        })
        .await
    }

    /// Ends an exact slot generation only after each published command has answered once.
    pub(super) fn finish(&self, tag: u64, result: Result<()>) -> Option<Waker> {
        let mut slot = self.slots[(tag / u64::from(DEPTH) % SLOTS as u64) as usize]
            .lock()
            .unwrap();
        let offset = tag % u64::from(DEPTH);
        let valid = slot.sequence == tag >> SEQUENCE_SHIFT && offset < slot.entries.len() as u64;
        if !valid || !matches!(slot.state, State::Active { .. }) {
            self.poison
                .set(Error::Device(format!("unexpected DMA completion {tag}")));
            return None;
        }
        self.receive(&mut slot, offset, result)
    }

    /// Records the answer to entry `offset` of the locked active `slot`, retiring the slot with
    /// its last.
    pub(super) fn receive(&self, slot: &mut Slot, offset: u64, result: Result<()>) -> Option<Waker> {
        let (count, bit) = (slot.entries.len(), 1u64 << offset);
        let State::Active { received, failure, .. } = &mut slot.state else {
            unreachable!("only a published slot is answered");
        };
        if *received & bit != 0 {
            let error = Error::Device("duplicate DMA completion".into());
            self.poison.set(error.clone());
            failure.get_or_insert(error);
            return None;
        }
        *received |= bit;
        // A command done is released now, not when its slot ends; a failed one stays for the
        // slot's failure to poison.
        match result {
            Ok(()) => slot.entries[offset as usize] = None,
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
        if received.count_ones() as usize != count {
            return None;
        }
        let observer = match failure.take() {
            Some(error) => {
                self.poison.set(error.clone());
                slot.fail(error)
            }
            None => slot.release(State::Complete(Ok(()))),
        };
        let waiters = self.waiters.lock().unwrap().take();
        waiters.wake();
        observer
    }
}

pub(super) struct Slot {
    pub(super) state: State,
    pub(super) waker: Option<Waker>,
    /// Each command until its engine reports it done, a failed one until the slot ends: a command
    /// stays at its address while in flight, since the kernel driver reads its descriptor in place.
    pub(super) entries: ArrayVec<Option<Route>, { DEPTH as usize }>,
    /// Whether a command moves host bytes the slot does not lease, which must not be freed before
    /// the slot ends.
    pub(super) borrowed: bool,
    pub(super) access: Lease,
    pub(super) sequence: u64,
}

impl Slot {
    /// Release before publication or after every submitted CQE has confirmed quiescence.
    pub(super) fn release(&mut self, state: State) -> Option<Waker> {
        self.access.clear();
        self.entries.clear();
        self.borrowed = false;
        self.state = state;
        self.waker.take()
    }

    /// Releases quiescent work with `error`, poisoning the host and device regions held for Write.
    pub(super) fn fail(&mut self, error: Error) -> Option<Waker> {
        self.access.poison();
        self.release(State::Complete(Err(error)))
    }
}

pub(super) enum State {
    Idle,
    Preparing,
    /// Submitted, waiting for room to publish its commands.
    Submitted,
    Active {
        /// The entries handed to their engine.
        published: u64,
        received: u64,
        failure: Option<Error>,
    },
    Complete(Result<()>),
}

/// The queue's transfer slots, each created on first use and only after every earlier one.
pub(super) struct Slots(Box<[OnceLock<Mutex<Slot>>]>);

impl Slots {
    pub(super) fn new(count: usize) -> Self {
        Self((0..count).map(|_| OnceLock::new()).collect())
    }

    /// The slots created so far.
    pub(super) fn created(&self) -> impl Iterator<Item = &Mutex<Slot>> {
        self.0.iter().map_while(OnceLock::get)
    }

    /// Every slot in order, created as the walk reaches it.
    fn each(&self) -> impl Iterator<Item = &Mutex<Slot>> {
        self.0.iter().map(|slot| {
            slot.get_or_init(|| {
                Mutex::new(Slot {
                    state: State::Idle,
                    waker: None,
                    entries: ArrayVec::new(),
                    borrowed: false,
                    access: Lease::new(DEPTH as usize * 2),
                    sequence: 0,
                })
            })
        })
    }
}

impl Index<usize> for Slots {
    type Output = Mutex<Slot>;

    fn index(&self, index: usize) -> &Mutex<Slot> {
        self.0[index].get().expect("a slot is created before it is used")
    }
}
