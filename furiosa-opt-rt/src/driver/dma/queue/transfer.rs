use std::future::poll_fn;
use std::task::Poll;
use std::time::{Duration, Instant};

use super::{DmaQueue, Route, State};
use crate::Result;
use crate::buffer::{Lease, Request};

/// A slot held by one caller; dropping it before submission releases its access.
pub(crate) struct Transfer<'a> {
    pub(super) queue: &'a DmaQueue,
    pub(super) index: usize,
    pub(super) sequence: u64,
}

impl Transfer<'_> {
    /// Holds `view` for this transfer's commands.
    pub(crate) fn access<'a>(&mut self, request: impl Into<Request<'a>>) -> Result<()> {
        let mut slot = self.queue.slots[self.index].lock().unwrap();
        slot.access.extend([request.into()])
    }

    pub(crate) fn hold(&mut self, access: Lease) {
        self.queue.slots[self.index].lock().unwrap().access = access;
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
            // A panic that poisoned the slot leaves its state whole, and this drop may run in its unwind.
            let mut slot = (self.queue.slots[self.index].lock()).unwrap_or_else(std::sync::PoisonError::into_inner);
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

    /// Copies a mailbox had no room for still end, with an error, once a mailbox failure stopped
    /// every cluster's mailbox.
    #[tokio::test]
    async fn ends_unsent_copies_after_mailboxes_stop() {
        let device = crate::test::device(&[0], 8);
        let host = crate::test::mapped(&device, vec![0], 4096);
        let registration = Arc::clone(host.registration().unwrap());
        let (mailbox, file) = crate::test::mailbox();
        let queue = DmaQueue::new(
            Transport::Fake(Fake::default()),
            Arc::new(std::sync::Mutex::new(vec![mailbox])),
            Waker::noop().clone(),
        );
        let mut transfer = queue.acquire().await.unwrap();
        for _ in 0..17 {
            let copy = crate::driver::dma::Copy::new(
                0,
                crate::driver::dma::Direction::Read,
                Arc::clone(&registration),
                0,
                0,
                0,
                1,
            );
            transfer.push(Route::Cluster(copy));
        }
        let mut submitted = Box::pin(transfer.submit());
        assert!(
            submitted
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        crate::test::respond(&file, &[(999, furiosa_opt_ipc::Answer(Ok(())))]);
        queue.complete();
        assert!(matches!(
            submitted.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(crate::Error::Device(_)))
        ));
        assert!(queue.complete());
    }

    /// Failed submission finishes its observer rather than leaving work permanently active.
    #[tokio::test]
    async fn completes_failed_submission() {
        let fake = Fake::default();
        fake.fail(crate::Error::Device("submission failed".into()));
        let queue = DmaQueue::new(Transport::Fake(fake), Arc::default(), Waker::noop().clone());
        let buffer = crate::test::allocator(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let host = crate::test::pinned();
        let mut transfer = queue.acquire().await.unwrap();
        push(&mut transfer, &buffer, Some(&host));
        let mut submitted = Box::pin(transfer.submit());
        assert!(matches!(
            submitted.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(crate::Error::Device(_))),
        ));
        assert!(queue.complete());
    }

    /// A failed host read makes its destination inaccessible after completion.
    #[tokio::test]
    async fn poisons_failed_destination() {
        let fake = Fake::default();
        let queue = DmaQueue::new(Transport::Fake(fake.clone()), Arc::default(), Waker::noop().clone());
        let buffer = crate::test::allocator(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let host = crate::test::pinned();
        let mut transfer = queue.acquire().await.unwrap();
        push(&mut transfer, &buffer, Some(&host));
        let tag = tag(&transfer);
        let mut submitted = Box::pin(transfer.submit());
        assert!(
            submitted
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        fake.respond(tag, Err(std::io::Error::other("DMA failed")));
        queue.complete();
        assert!(std::panic::catch_unwind(|| host[0]).is_err());
    }

    /// A failed host read does not modify or poison its device source.
    #[tokio::test]
    async fn preserves_failed_source() {
        let fake = Fake::default();
        let queue = DmaQueue::new(Transport::Fake(fake.clone()), Arc::default(), Waker::noop().clone());
        let buffer = crate::test::allocator(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let host = crate::test::pinned();
        let mut transfer = queue.acquire().await.unwrap();
        push(&mut transfer, &buffer, Some(&host));
        let tag = tag(&transfer);
        let mut submitted = Box::pin(transfer.submit());
        assert!(
            submitted
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        fake.respond(tag, Err(std::io::Error::other("DMA failed")));
        queue.complete();
        assert_eq!(buffer.slice(..).unwrap().ready(crate::Access::Read).await, Ok(()));
    }

    /// A failed host write makes its device destination inaccessible after completion.
    #[tokio::test]
    async fn poisons_device_destination() {
        let fake = Fake::default();
        let queue = DmaQueue::new(Transport::Fake(fake.clone()), Arc::default(), Waker::noop().clone());
        let buffer = crate::test::allocator(0..512, 512)
            .alloc(crate::memory::Kind::Dram, 1)
            .unwrap();
        let host = crate::test::pinned();
        let mut transfer = queue.acquire().await.unwrap();
        transfer
            .access((buffer.slice(..).unwrap(), crate::Access::Write))
            .unwrap();
        transfer.push(crate::driver::dma::Route::Channel(crate::driver::dma::Command::new(
            0,
            crate::driver::dma::Direction::Write,
            crate::driver::dma::Host::Owned(Arc::clone(host.dmabuf()), 0),
            0,
            1,
        )));
        let tag = tag(&transfer);
        let mut submitted = Box::pin(transfer.submit());
        assert!(
            submitted
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        fake.respond(tag, Err(std::io::Error::other("DMA failed")));
        queue.complete();
        assert_eq!(
            buffer.slice(..).unwrap().ready(crate::Access::Read).await,
            Err(crate::Error::Poisoned)
        );
        assert_eq!(host[0], 0);
    }

    /// A split read excludes host readers until its last DMA command completes.
    #[tokio::test]
    async fn excludes_reads_until_last_completion() {
        let fake = Fake::default();
        let queue = DmaQueue::new(Transport::Fake(fake.clone()), Arc::default(), Waker::noop().clone());
        let host = crate::test::pinned();
        let backing = host.slice(..).pinned.unwrap();
        let mut transfer = queue.acquire().await.unwrap();
        transfer
            .access(Request::host(backing, 2, crate::Access::Write))
            .unwrap();
        for offset in 0..2 {
            transfer.push(crate::driver::dma::Route::Channel(crate::driver::dma::Command::new(
                0,
                crate::driver::dma::Direction::Read,
                crate::driver::dma::Host::Owned(Arc::clone(backing.dmabuf), offset),
                0,
                1,
            )));
        }
        let tag = tag(&transfer);
        let mut submitted = Box::pin(transfer.submit());
        assert!(
            submitted
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        fake.respond(tag, Ok(1));
        queue.complete();
        assert!(matches!(
            Lease::new(1).extend([Request::host(backing, 2, crate::Access::Read)]),
            Err(crate::Error::Busy)
        ));
        fake.respond(tag + 1, Ok(1));
        queue.complete();
        assert!(
            Lease::new(1)
                .extend([Request::host(backing, 2, crate::Access::Read)])
                .is_ok()
        );
    }

    /// A transfer of borrowed host bytes completes before its first poll returns, and dropping it
    /// submitted, as a panic unwinding through its poll does, returns only once the device answered.
    #[tokio::test]
    async fn completes_borrowed_slices_before_returning() {
        let buffer = crate::test::allocator(0..512, 512)
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
            let buffer = crate::test::allocator(0..512, 512)
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
