use std::future::poll_fn;
use std::sync::Mutex;
use std::task::{Poll, Waker};

use arrayvec::ArrayVec;

use super::{DEPTH, DmaQueue, Route, SEQUENCE_SHIFT, SLOTS, Transfer};
use crate::buffer::Holds;
use crate::{Error, Result};

pub(super) struct Slot {
    pub(super) state: State,
    pub(super) waker: Option<Waker>,
    pub(super) entries: ArrayVec<Route, { DEPTH as usize }>,
    pub(super) access: Holds,
    pub(super) sequence: u64,
}

pub(super) enum State {
    Idle,
    Preparing,
    /// Submitted, waiting for room to publish its commands.
    Submitted,
    Active {
        received: u64,
        failure: Option<Error>,
    },
    Complete(Result<()>),
}

impl DmaQueue {
    /// Retires an exact slot generation only after each published command has answered once.
    pub(super) fn complete(&self, tag: u64, result: Result<()>) -> Option<Waker> {
        let mut slot = self
            .slot((tag / u64::from(DEPTH) % SLOTS as u64) as usize)
            .lock()
            .unwrap();
        let offset = tag % u64::from(DEPTH);
        let valid = slot.sequence == tag >> SEQUENCE_SHIFT && offset < slot.entries.len() as u64;
        if !valid || !matches!(slot.state, State::Active { .. }) {
            self.failure
                .get_or_init(|| Error::Device(format!("unexpected DMA completion {tag}")));
            return None;
        }
        self.receive(&mut slot, offset, result)
    }

    /// Records the answer to entry `offset` of the locked active `slot`, retiring the slot with
    /// its last.
    pub(super) fn receive(&self, slot: &mut Slot, offset: u64, result: Result<()>) -> Option<Waker> {
        let (count, bit) = (slot.entries.len(), 1u64 << offset);
        let State::Active { received, failure } = &mut slot.state else {
            unreachable!("only a published slot is answered");
        };
        if *received & bit != 0 {
            let error = Error::Device("duplicate DMA completion".into());
            self.failure.get_or_init(|| error.clone());
            failure.get_or_insert(error);
            return None;
        }
        *received |= bit;
        if let Err(error) = result {
            failure.get_or_insert(error);
        }
        if received.count_ones() as usize != count {
            return None;
        }
        let result = failure.take().map_or(Ok(()), Err);
        if let Err(error) = &result {
            self.failure.get_or_init(|| error.clone());
        }
        let observer = slot.release(State::Complete(result));
        let waiters = self.waiters.lock().unwrap().take();
        waiters.wake();
        observer
    }

    pub(crate) async fn acquire(&self) -> Result<Transfer<'_>> {
        poll_fn(|context| {
            if let Some(error) = self.failure.get() {
                return Poll::Ready(Err(error.clone()));
            }
            for (index, slot) in self.slots.iter().enumerate() {
                let mut slot = slot
                    .get_or_init(|| {
                        Mutex::new(Slot {
                            state: State::Idle,
                            waker: None,
                            entries: ArrayVec::new(),
                            access: Holds::new(DEPTH as usize),
                            sequence: 0,
                        })
                    })
                    .lock()
                    .unwrap();
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
}

impl Slot {
    /// Release before publication or after every submitted CQE has confirmed quiescence.
    pub(super) fn release(&mut self, state: State) -> Option<Waker> {
        self.access.clear();
        self.entries.clear();
        self.state = state;
        self.waker.take()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::task::Context;

    use super::super::fixture::{push, tag};
    use super::*;
    use crate::Access;
    use crate::driver::dma::{Transport, fake::Fake};

    /// Only a transport answer releases DMA memory; driver errors reach the observer.
    #[tokio::test]
    async fn retains_memory_until_transport_answers() {
        for fails in [false, true] {
            let buffer = crate::test::allocations(0..512, 512)
                .alloc(crate::memory::Kind::Dram, 1)
                .unwrap();
            let host = crate::test::pinned();
            let fake = Fake::default();
            let queue = DmaQueue::new(Transport::Fake(fake.clone()), Arc::default(), Waker::noop().clone());
            let mut transfer = queue.acquire().await.unwrap();
            push(&mut transfer, &buffer, Some(&host));
            let tag = tag(&transfer);
            let mut observer = std::pin::pin!(transfer.submit());
            let mut context = Context::from_waker(Waker::noop());
            assert_eq!(observer.as_mut().poll(&mut context), Poll::Pending);
            let mut access = Holds::new(1);
            let view = buffer.slice(..).unwrap();
            assert_eq!(access.acquire([(view, Access::Write)]), Err(Error::Busy));

            let result = if fails {
                Err(std::io::Error::from_raw_os_error(nix::libc::EIO))
            } else {
                Ok(1)
            };
            fake.respond(tag, result);
            assert!(queue.progress());

            access.acquire([(view, Access::Write)]).unwrap();
            let result = observer.as_mut().poll(&mut context);
            if fails {
                assert!(matches!(result, Poll::Ready(Err(Error::Device(_)))));
            } else {
                assert_eq!(result, Poll::Ready(Ok(())));
            }
        }
    }

    /// Far more transfers than a device's channels stay in flight at once, each submitted
    /// without waiting for an earlier one.
    #[tokio::test]
    async fn keeps_many_transfers_in_flight() {
        let buffer = crate::test::allocations(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let host = crate::test::pinned();
        let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
        let mut context = Context::from_waker(Waker::noop());
        let mut observers = Vec::new();
        for _ in 0..100 {
            let mut transfer = queue.acquire().await.unwrap();
            push(&mut transfer, &buffer, Some(&host));
            let tag = tag(&transfer);
            let mut observer = Box::pin(transfer.submit());
            assert_eq!(observer.as_mut().poll(&mut context), Poll::Pending);
            observers.push((tag, observer));
        }
        for (tag, mut observer) in observers {
            queue.complete(tag, Ok(()));
            assert_eq!(observer.as_mut().poll(&mut context), Poll::Ready(Ok(())));
        }
    }

    /// A transfer holds its device and host memory until every answer, observed or not.
    #[tokio::test]
    async fn retains_access_after_observer_loss() {
        for forget in [false, true] {
            for fails in [false, true] {
                let buffer = crate::test::allocations(0..512, 512)
                    .alloc(crate::memory::Kind::Dram, 1)
                    .unwrap();
                let hosts = [crate::test::pinned(), crate::test::pinned()];
                let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
                let mut transfer = queue.acquire().await.unwrap();
                hosts.iter().for_each(|host| push(&mut transfer, &buffer, Some(host)));
                let (tag, slot) = (tag(&transfer), queue.slot(transfer.index));
                let mut observer = Box::pin(transfer.submit());
                assert_eq!(
                    observer.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                    Poll::Pending
                );
                let mut access = Holds::new(1);
                let view = buffer.slice(..).unwrap();
                let failure = fails.then(|| Error::Device("failed DMA".into()));
                slot.lock().unwrap().state = State::Active { received: 0, failure };
                if forget {
                    std::mem::forget(observer);
                } else {
                    drop(observer);
                }
                // One of two answers, or a duplicate of it, does not release the slot.
                for _ in 0..1 + usize::from(fails) {
                    assert!(queue.complete(tag + 1, Ok(())).is_none());
                    assert_eq!(access.acquire([(view, Access::Write)]), Err(Error::Busy));
                }
                queue.complete(tag, Ok(()));
                access.acquire([(view, Access::Write)]).unwrap();
                assert_eq!(hosts[0][0], 0);
                assert_eq!(queue.acquire().await.is_err(), fails);
            }
        }
    }
}
