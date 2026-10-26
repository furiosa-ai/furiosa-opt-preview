//! io_uring submission and completion for driver commands.

use arrayvec::ArrayVec;
use io_uring::{IoUring, squeue::Entry};
use std::io;
use std::time::Duration;

use super::engine::Engine;
use crate::{Error, Result};

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
}

impl Engine for Ring {
    type Command = Entry;

    /// Queues every command for the kernel's poller without waiting, or none when the batch does
    /// not fit. Commands a failed wake left queued wait for [`Self::complete`] to wake it again.
    fn submit<'a>(&mut self, commands: impl Iterator<Item = (u64, &'a Entry)>) -> Result<usize> {
        let entries = (commands.map(|(tag, entry)| entry.clone().user_data(tag)))
            .collect::<ArrayVec<_, { super::queue::DEPTH as usize }>>();
        if entries.is_empty() {
            return Ok(0);
        }
        let mut queue = self.0.submission();
        // SAFETY: the caller retains every referenced descriptor until completion; push_multiple
        // checks capacity before publishing any entry.
        if unsafe { queue.push_multiple(&entries) }.is_err() {
            return Ok(0);
        }
        drop(queue);
        // The poller takes whatever the queue holds; this call only wakes it when it went idle,
        // and the count it returns is the queue's length then, not a batch to wait for. The
        // entries are published already, so no error may make the caller submit them again.
        let _ = self.0.submit();
        Ok(entries.len())
    }

    /// Hands `done` every completion that arrived, with the driver's result; an error when the
    /// poller cannot be woken for the commands still queued.
    fn complete(&mut self, mut done: impl FnMut(u64, Result<()>)) -> Result<()> {
        // Commands still queued after a failed wake wait for another one; with the poller awake,
        // this enters nothing.
        let woken = if self.0.submission().is_empty() {
            Ok(())
        } else {
            (self.0.submit().map(drop)).map_err(|error| Error::Device(format!("waking the DMA ring: {error}")))
        };
        // Completions that found the queue full, and every one after them, wait in the kernel
        // until an enter collects them, which the submission poller never does.
        if self.0.submission().cq_overflow() {
            // SAFETY: an enter that submits nothing and passes no argument only collects; one that
            // fails leaves the flag set, so the next collection enters again.
            let _ = unsafe { self.0.submitter().enter::<()>(0, 0, ENTER_GETEVENTS, None) };
        }
        for entry in self.0.completion() {
            let result = match entry.result() {
                errno if errno < 0 => Err(Error::Device(format!(
                    "DMA command failed: {}",
                    io::Error::from_raw_os_error(-errno)
                ))),
                _ => Ok(()),
            };
            done(entry.user_data(), result);
        }
        woken
    }
}

/// How long the kernel's submission poller keeps spinning after the last command.
const SQPOLL_IDLE: Duration = Duration::from_millis(100);
/// Completion slots per submission slot: commands the poller takes stay in flight beyond the
/// submission queue, and each one completes into this queue.
const CQ_PER_SQ: u32 = 64;
/// `io_uring_enter` flag that collects completions, moving overflowed ones into the queue.
const ENTER_GETEVENTS: u32 = 1;

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use io_uring::IoUring;
    use io_uring::opcode::Nop;

    use super::{Engine, Ring};

    /// A command published before its wake failed still answers with its own result, and the
    /// ring reports the failure until the poller can be woken.
    #[test]
    fn answers_after_failed_wake() {
        // A disabled ring refuses every enter, so the wake of its sleeping poller fails.
        let ring = IoUring::builder().setup_sqpoll(1).setup_r_disabled().build(1).unwrap();
        let mut ring = Ring(ring);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ring.0.submission().need_wakeup() {
            assert!(Instant::now() < deadline, "the poller never slept");
        }
        // The poller sets the flag just before it sleeps.
        std::thread::sleep(Duration::from_millis(20));
        let nop = Nop::new().build();
        assert_eq!(ring.submit(std::iter::once((7, &nop))).unwrap(), 1);
        let mut answered = Vec::new();
        assert!(ring.complete(|tag, result| answered.push((tag, result))).is_err());
        ring.0.submitter().register_enable_rings().unwrap();
        while answered.is_empty() && Instant::now() < deadline {
            let _ = ring.complete(|tag, result| answered.push((tag, result)));
        }
        assert!(matches!(answered.as_slice(), [(7, Ok(()))]), "{answered:?}");
    }

    /// Every command is answered once, however many answers arrive before the first collection.
    #[test]
    fn answers_commands_collected_late() {
        // One submission slot, so far more commands complete than the completion queue holds.
        let mut ring = Ring::open(1).unwrap();
        let commands = 256;
        for tag in 0..commands {
            let nop = Nop::new().build().user_data(tag);
            while ring.submit(std::iter::once((tag, &nop))).unwrap() == 0 {}
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut answered = Vec::new();
        while answered.len() < commands as usize && Instant::now() < deadline {
            ring.complete(|tag, result| answered.push((tag, result.is_ok())))
                .unwrap();
        }
        answered.sort_unstable();
        let expected = (0..commands).map(|tag| (tag, true));
        assert!(
            answered.iter().copied().eq(expected),
            "{} of {commands} answered",
            answered.len()
        );
    }
}
