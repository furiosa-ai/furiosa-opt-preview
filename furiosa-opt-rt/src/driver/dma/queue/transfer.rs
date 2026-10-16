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
        let mut slot = self.queue.slot(self.index).lock().unwrap();
        slot.access.acquire([(view, access)])
    }

    pub(crate) fn push(&mut self, route: Route) {
        let mut slot = self.queue.slot(self.index).lock().unwrap();
        slot.entries.push(route);
    }

    pub(crate) async fn submit(self) -> Result<()> {
        let borrowed = {
            let slot = self.queue.slot(self.index).lock().unwrap();
            if slot.entries.is_empty() {
                return Ok(());
            }
            slot.entries.iter().any(Route::borrowed)
        };
        if let Some(error) = self.queue.failure.get() {
            return Err(error.clone());
        }
        // The queue owns the slot from here: whoever polls next publishes it once there is room.
        self.queue.slot(self.index).lock().unwrap().state = State::Submitted;
        self.queue.waiting.lock().unwrap().push_back(self.index);
        if !self.queue.publish_waiting() || borrowed {
            self.queue.worker.wake_by_ref();
        }
        let started = Instant::now();
        poll_fn(|context| {
            loop {
                // Unmanaged slices cannot outlive this poll, even if the caller forgets its future.
                let spinning = !borrowed && started.elapsed() < Duration::from_micros(50);
                if borrowed || spinning {
                    self.queue.progress();
                }
                let mut slot = self.queue.slot(self.index).lock().unwrap();
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
            let mut slot = self.queue.slot(self.index).lock().unwrap();
            if slot.sequence != self.sequence {
                return;
            }
            match slot.state {
                State::Preparing => {
                    slot.release(State::Idle);
                    return;
                }
                State::Submitted | State::Active { .. } if slot.entries.iter().any(Route::borrowed) => {}
                _ => return,
            }
            drop(slot);
            self.queue.progress();
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, OnceLock, Weak};
    use std::task::{Context, Waker};

    use super::super::{
        SEQUENCE_SHIFT,
        fixture::{push, tag},
    };
    use super::*;
    use crate::Error;
    use crate::driver::dma::{Transport, fake::Fake};

    #[tokio::test]
    async fn notifies_registered_observer() {
        struct Observer(std::sync::atomic::AtomicUsize);
        impl std::task::Wake for Observer {
            fn wake(self: Arc<Self>) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        struct Worker(OnceLock<Weak<DmaQueue>>);
        impl std::task::Wake for Worker {
            fn wake(self: Arc<Self>) {
                let queue = self.0.get().unwrap().upgrade().unwrap();
                let tag = queue.slot(0).lock().unwrap().sequence << SEQUENCE_SHIFT;
                queue.complete(tag, Ok(())).unwrap().wake();
            }
        }
        let buffer = crate::test::allocations(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let host = crate::test::pinned();
        let worker = Arc::new(Worker(OnceLock::new()));
        let queue = Arc::new(DmaQueue::new(
            Transport::Fake(Fake::default()),
            Arc::default(),
            worker.clone().into(),
        ));
        assert!(worker.0.set(Arc::downgrade(&queue)).is_ok());
        let mut transfer = queue.acquire().await.unwrap();
        push(&mut transfer, &buffer, Some(&host));
        let mut observer = std::pin::pin!(transfer.submit());
        let wakes = Arc::new(Observer(std::sync::atomic::AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&wakes));
        let mut context = Context::from_waker(&waker);
        assert_eq!(observer.as_mut().poll(&mut context), Poll::Pending);
        assert_eq!(wakes.0.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(observer.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    }

    /// A transfer of borrowed host bytes completes before its first poll returns.
    #[test]
    fn completes_borrowed_slices_before_returning() {
        let buffer = crate::test::allocations(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
        std::thread::scope(|scope| {
            scope.spawn(|| {
                loop {
                    let mut slot = queue.slot(0).lock().unwrap();
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
    }

    /// Dropping a submitted transfer of unmanaged host bytes, as a panic unwinding through its
    /// poll does, returns only once the device has answered it.
    #[tokio::test]
    async fn waits_for_borrowed_slices_when_dropped() {
        let buffer = crate::test::allocations(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
        let mut transfer = queue.acquire().await.unwrap();
        push(&mut transfer, &buffer, None);
        let tag = tag(&transfer);
        queue.slot(transfer.index).lock().unwrap().state = State::Active {
            received: 0,
            failure: None,
        };
        let answered = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(20));
                answered.store(true, std::sync::atomic::Ordering::Release);
                queue.complete(tag, Ok(()));
            });
            drop(transfer);
            assert!(
                answered.load(std::sync::atomic::Ordering::Acquire),
                "the drop waited for the answer"
            );
        });
    }

    /// A transfer's result survives its slot's reuse, and a failure stops the queue.
    #[tokio::test]
    async fn preserves_completed_results() {
        for result in [Ok(()), Err(Error::Poisoned)] {
            let buffer = crate::test::allocations(0..512, 512)
                .alloc(crate::memory::Kind::Dram, 1)
                .unwrap();
            let host = crate::test::pinned();
            let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
            let mut transfer = queue.acquire().await.unwrap();
            push(&mut transfer, &buffer, Some(&host));
            let tag = tag(&transfer);
            let mut observer = std::pin::pin!(transfer.submit());
            let mut context = Context::from_waker(Waker::noop());
            assert_eq!(observer.as_mut().poll(&mut context), Poll::Pending);
            assert!(
                queue.complete(tag, result.clone()).is_some(),
                "completion wakes the observer"
            );
            // A reused slot answers its earlier observer with success; a failure rejects reuse.
            let next = queue.acquire().await;
            assert_eq!(next.is_ok(), result.is_ok());
            assert!(queue.complete(tag, Ok(())).is_none());
            assert!(queue.failure.get().is_some());
            assert_eq!(observer.as_mut().poll(&mut context), Poll::Ready(result));
        }
    }
}
