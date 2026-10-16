//! The thread that drives admitted work to completion while nobody observes it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Wake, Waker};
use std::thread::JoinHandle;
use std::thread::Thread;
use std::time::Duration;

use crate::driver::Driver;
use crate::function::Launches;

/// How often an idle worker looks for progress nobody woke it for.
const IDLE_POLL: Duration = Duration::from_millis(1);

/// Each device's thread: it spins on a core while any launch or transfer runs, so completions are
/// seen at once, and wakes every [`IDLE_POLL`] while idle.
pub(crate) struct Worker {
    /// Every function's launch slots, so launches nobody polls still retire and run their epilogues.
    pub(crate) launches: Arc<Launches>,
    /// Wakes the thread from its idle wait when work it must retire was published.
    pub(crate) waker: Waker,
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// Unparks the worker once it runs; a wake before that finds it starting, not idle.
#[derive(Default)]
pub(crate) struct Unpark(OnceLock<Thread>);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.get().map(Thread::unpark);
    }
}

impl Worker {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self {
            launches: Arc::default(),
            waker: Waker::noop().clone(),
            shutdown: Arc::default(),
            thread: None,
        }
    }

    /// Drives `driver`'s work; `unpark` must be the one behind the waker the driver was opened
    /// with.
    pub(crate) fn spawn(driver: Arc<Driver>, unpark: Arc<Unpark>) -> Self {
        let launches = Arc::new(Launches::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        let (active, stopped) = (Arc::clone(&launches), Arc::clone(&shutdown));
        let thread = std::thread::spawn(move || {
            loop {
                let finished = active.progress();
                let failed = driver.failure.get().is_some();
                if driver.transfers.progress() && (finished || failed) {
                    if stopped.load(Ordering::Acquire) {
                        // Stop task code before releasing its DMA channels or peer mappings.
                        driver.stop();
                        return;
                    }
                    std::thread::park_timeout(IDLE_POLL);
                }
                std::hint::spin_loop();
            }
        });
        unpark.0.get_or_init(|| thread.thread().clone());
        Self {
            launches,
            waker: Waker::from(unpark),
            shutdown,
            thread: Some(thread),
        }
    }
}

impl Drop for Worker {
    /// The worker drains admitted work before the device releases its resources.
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.waker.wake_by_ref();
        // The last Device can drop inside an epilogue on the worker; it stops on `shutdown`.
        if let Some(thread) = self.thread.take()
            && thread.thread().id() != std::thread::current().id()
        {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dropping the last `Device` returns only once its worker has stopped.
    #[test]
    fn shutdown_waits() {
        let (device, _) = crate::test::running(&[0], 8);
        let links = device.driver.links.lock().unwrap();
        let worker = Worker::spawn(Arc::clone(&device.driver), Arc::default());
        let shutdown = Arc::clone(&worker.shutdown);
        let (done, complete) = std::sync::mpsc::sync_channel(1);
        let dropping = std::thread::spawn(move || {
            drop(worker);
            done.send(()).unwrap();
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !shutdown.load(Ordering::Acquire) {
            assert!(std::time::Instant::now() < deadline, "shutdown was not requested");
            std::thread::yield_now();
        }
        assert!(matches!(complete.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)));
        drop(links);
        complete
            .recv_timeout(Duration::from_secs(2))
            .expect("drop joined the worker");
        dropping.join().unwrap();
        assert!(device.driver.links.lock().unwrap().is_empty());
    }
}
