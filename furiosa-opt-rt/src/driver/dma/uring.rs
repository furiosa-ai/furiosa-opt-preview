//! io_uring submission and completion for driver commands.

use io_uring::{IoUring, squeue::Entry};
use std::io;
use std::time::Duration;

/// How long the kernel's submission poller keeps spinning after the last command.
const SQPOLL_IDLE: Duration = Duration::from_millis(100);
/// Completion slots per submission slot: commands the poller takes stay in flight beyond the
/// submission queue, and each one completes into this queue.
const CQ_PER_SQ: u32 = 64;
/// `io_uring_enter` flag that collects completions, moving overflowed ones into the queue.
const ENTER_GETEVENTS: u32 = 1;

pub(crate) struct Ring(IoUring);

impl Ring {
    /// A ring that queues `depth` commands for the kernel poller, which takes them without system
    /// calls until [`SQPOLL_IDLE`] of quiet.
    pub(crate) fn open(depth: u32) -> io::Result<Self> {
        IoUring::builder()
            .setup_sqpoll(SQPOLL_IDLE.as_millis() as u32)
            .setup_cqsize(depth * CQ_PER_SQ)
            .build(depth)
            .map(Self)
    }

    /// Queues every command for the kernel's poller without waiting: `WouldBlock` with nothing
    /// queued when the batch does not fit; any other error means the ring itself is broken.
    pub(crate) fn submit(&mut self, entries: &[Entry]) -> io::Result<()> {
        let mut queue = self.0.submission();
        // SAFETY: the caller retains every referenced descriptor until completion; push_multiple
        // checks capacity before publishing any entry.
        unsafe { queue.push_multiple(entries) }.map_err(|_| std::io::ErrorKind::WouldBlock)?;
        drop(queue);
        // The poller takes whatever the queue holds; this call only wakes it when it went idle,
        // and the count it returns is the queue's length then, not a batch to wait for.
        self.0.submit().map(drop).map_err(|error| {
            // Entries are published already; WouldBlock must never authorize resubmitting the batch.
            if error.kind() == std::io::ErrorKind::WouldBlock {
                std::io::Error::other(error)
            } else {
                error
            }
        })
    }

    /// Every completion that has arrived: its tag and the driver's result, an `errno` when negative.
    pub(crate) fn completions(&mut self) -> impl Iterator<Item = (u64, io::Result<i32>)> + '_ {
        // Completions that found the queue full, and every one after them, wait in the kernel
        // until an enter collects them, which the submission poller never does.
        if self.0.submission().cq_overflow() {
            // SAFETY: an enter that submits nothing and passes no argument only collects; one that
            // fails leaves the flag set, so the next collection enters again.
            let _ = unsafe { self.0.submitter().enter::<()>(0, 0, ENTER_GETEVENTS, None) };
        }
        self.0.completion().map(|entry| {
            let result = entry.result();
            let result = if result < 0 {
                Err(std::io::Error::from_raw_os_error(-result))
            } else {
                Ok(result)
            };
            (entry.user_data(), result)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use io_uring::opcode::Nop;

    use super::Ring;

    /// Every command is answered once, however many answers arrive before the first collection.
    #[test]
    fn answers_commands_collected_late() {
        // One submission slot, so far more commands complete than the completion queue holds.
        let mut ring = Ring::open(1).unwrap();
        let commands = 256;
        for tag in 0..commands {
            let nop = Nop::new().build().user_data(tag);
            while let Err(error) = ring.submit(std::slice::from_ref(&nop)) {
                assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock, "{error}");
            }
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut answered = Vec::new();
        while answered.len() < commands as usize && Instant::now() < deadline {
            answered.extend(ring.completions().map(|(tag, result)| (tag, result.unwrap())));
        }
        answered.sort_unstable();
        let expected = (0..commands).map(|tag| (tag, 0));
        assert!(
            answered.iter().copied().eq(expected),
            "{} of {commands} answered",
            answered.len()
        );
    }
}
