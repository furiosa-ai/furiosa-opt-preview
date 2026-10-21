use std::future::poll_fn;
use std::task::Poll;
use std::time::{Duration, Instant};

use super::{DmaQueue, Route, State};
use crate::{Access, BufferSlice, Result};

/// A slot held by one caller; dropping it before submission releases its access.
pub(crate) struct Transfer<'a> {
    pub(super) queue: &'a DmaQueue,
    pub(super) index: usize,
    pub(super) sequence: u64,
}

impl Transfer<'_> {
    /// Holds `view` for this transfer's commands.
    pub(crate) fn access(&mut self, view: BufferSlice<'_>, access: Access) -> Result<()> {
        let mut slot = self.queue.slots[self.index].lock().unwrap();
        slot.access.extend([(view, access)])
    }

    pub(crate) fn push(&mut self, route: Route) {
        let mut slot = self.queue.slots[self.index].lock().unwrap();
        slot.borrowed |= route.borrowed();
        slot.entries.push(Some(route));
    }

    pub(crate) async fn submit(self) -> Result<()> {
        let borrowed = {
            let slot = self.queue.slots[self.index].lock().unwrap();
            if slot.entries.is_empty() {
                return Ok(());
            }
            slot.borrowed
        };
        if let Some(error) = self.queue.poison.get() {
            return Err(error.clone());
        }
        // The queue owns the slot from here: whoever polls next publishes it once there is room.
        self.queue.slots[self.index].lock().unwrap().state = State::Submitted;
        self.queue.waiting.lock().unwrap().push_back(self.index);
        self.queue.complete();
        let published = !matches!(self.queue.slots[self.index].lock().unwrap().state, State::Submitted);
        if !published || borrowed {
            self.queue.worker.wake_by_ref();
        }
        let started = Instant::now();
        poll_fn(|context| {
            loop {
                // Unmanaged slices cannot outlive this poll, even if the caller forgets its future.
                let spinning = !borrowed && started.elapsed() < Duration::from_micros(50);
                if borrowed || spinning {
                    self.queue.complete();
                }
                let mut slot = self.queue.slots[self.index].lock().unwrap();
                if slot.sequence != self.sequence {
                    return Poll::Ready(Ok(()));
                }
                if let State::Complete(result) = &slot.state {
                    return Poll::Ready(result.clone());
                }
                if !borrowed && !spinning {
                    slot.waker
                        .get_or_insert_with(|| context.waker().clone())
                        .clone_from(context.waker());
                    drop(slot);
                    self.queue.worker.wake_by_ref();
                    return Poll::Pending;
                }
                drop(slot);
                if borrowed {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
            }
        })
        .await
    }
}

impl Drop for Transfer<'_> {
    /// An unpublished slot is released. A submitted one that copies unmanaged host bytes is waited
    /// for, so a panic unwinding through its poll cannot free bytes the device still copies.
    fn drop(&mut self) {
        loop {
            let mut slot = self.queue.slots[self.index].lock().unwrap();
            if slot.sequence != self.sequence {
                return;
            }
            match slot.state {
                State::Preparing => {
                    slot.release(State::Idle);
                    return;
                }
                State::Submitted | State::Active { .. } if slot.borrowed => {}
                _ => return,
            }
            drop(slot);
            self.queue.complete();
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::task::{Context, Waker};

    use super::super::fixture::{push, tag};
    use super::*;
    use crate::driver::dma::{Transport, fake::Fake};

    /// A transfer of borrowed host bytes completes before its first poll returns, and dropping it
    /// submitted, as a panic unwinding through its poll does, returns only once the device answered.
    #[tokio::test]
    async fn completes_borrowed_slices_before_returning() {
        let buffer = crate::test::allocations(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
        std::thread::scope(|scope| {
            scope.spawn(|| {
                loop {
                    let mut slot = queue.slots[0].lock().unwrap();
                    if matches!(slot.state, State::Active { .. }) {
                        break slot.release(State::Complete(Ok(())));
                    }
                }
            });
            let submit = std::pin::pin!(async {
                let mut transfer = queue.acquire().await?;
                push(&mut transfer, &buffer, None);
                transfer.submit().await
            });
            assert_eq!(
                submit.poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Ok(()))
            );
        });

        // Dropping the submitted transfer waits for the device's answer.
        {
            let buffer = crate::test::allocations(0..512, 512)
                .alloc(crate::memory::Kind::Dram, 1)
                .unwrap();
            let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
            let mut transfer = queue.acquire().await.unwrap();
            push(&mut transfer, &buffer, None);
            let tag = tag(&transfer);
            queue.slots[transfer.index].lock().unwrap().state = State::Active {
                published: u64::MAX,
                received: 0,
                failure: None,
            };
            let answered = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    std::thread::sleep(Duration::from_millis(20));
                    answered.store(true, std::sync::atomic::Ordering::Release);
                    queue.finish(tag, Ok(()));
                });
                drop(transfer);
                assert!(
                    answered.load(std::sync::atomic::Ordering::Acquire),
                    "the drop waited for the answer"
                );
            });
        }
    }
}
