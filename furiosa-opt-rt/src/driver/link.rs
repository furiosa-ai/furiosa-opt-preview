//! Host request correlation and protocol encoding over arch word queues.

use super::cluster::Cluster;
use crate::{Error, Result};
use furiosa_opt_abi::bootloader;
use furiosa_opt_abi::ring::{Completion, header};
use furiosa_opt_ipc::{LAUNCH_TIMEOUT, MAX_SUBMISSION_WORDS, Request, Response, Status, wire};
use std::task::Poll;
use std::time::{Duration, Instant};

/// How long the host waits for an answer: past [`LAUNCH_TIMEOUT`], so a launch the device timed out still
/// answers first, and short of the driver's own doorbell deadline, 60 s by default.
pub(crate) const PATIENCE: Duration = LAUNCH_TIMEOUT.saturating_add(Duration::from_secs(2));

pub(crate) struct Link {
    pub(crate) queues: Cluster,
    next: u32,
    answers: [Answer; IN_FLIGHT],
}

/// Requests one link keeps in flight at most; a submitter past this many waits for an answer
/// to be collected.
pub(crate) const IN_FLIGHT: usize = 16;

/// Reply storage grows for diagnostics; ordinary completions reuse its capacity.
#[derive(Default)]
struct Answer {
    id: Option<u32>,
    words: Vec<u64>,
}

impl Answer {
    /// The response, once every completion carrying it has arrived.
    fn decode(&self) -> std::result::Result<Option<Response>, furiosa_opt_ipc::Error> {
        match wire::decode(&self.words) {
            Err(furiosa_opt_ipc::Error::Truncated) => Ok(None),
            response => response.map(Some),
        }
    }
}

pub(crate) struct Prepared<'a> {
    publication: furiosa_opt_abi::ring::Publication<'a>,
    answer: &'a mut Answer,
    next: &'a mut u32,
}

impl Prepared<'_> {
    pub(crate) fn publish(self) -> u32 {
        let id = *self.next;
        self.answer.id = Some(id);
        *self.next = id.wrapping_add(1);
        self.publication.publish();
        id
    }
}

impl Link {
    pub(crate) fn new(queues: Cluster) -> Self {
        Self {
            queues,
            next: 1,
            answers: Default::default(),
        }
    }

    fn receive(&mut self) -> Result<Option<Completion>> {
        let queue = &self.queues.completion;
        let mut words = [0; _];
        match queue.copy(0, &mut words).and_then(|()| queue.consume(words.len())) {
            Err(furiosa_opt_abi::ring::Error::Words) => return Ok(None),
            result => result.map_err(|error| Error::Device(error.to_string()))?,
        }
        let invalid = || Error::Device(format!("invalid completion header {:#x}", words[0]));
        Completion::decode(words).map(Some).ok_or_else(invalid)
    }

    /// Starts the staged firmware through the bootloader contract.
    pub(crate) fn boot(&mut self, addr: u64, len: u64, token: u64) -> Result<bootloader::Code> {
        let id = self.next;
        self.next = id.wrapping_add(1);
        let frame = bootloader::Request { addr, len, token }.encode(id);
        let Poll::Ready(result) = self.queues.submission.write_all(&frame) else {
            return Err(Error::Device("the boot submission queue is full".into()));
        };
        result.map_err(|why| Error::Device(format!("submitting the boot request: {why}")))?;
        let completion = self.wait(Self::receive)?;
        let (answered, code) = (completion.id, completion.payload[0]);
        if answered != id {
            return Err(Error::Device(format!(
                "boot completion {answered} does not answer {id}"
            )));
        }
        bootloader::Code::try_from(code).map_err(|code| Error::Device(format!("the bootloader answered with {code}")))
    }

    /// Spins until `step` yields, failing once the device takes longer than [`PATIENCE`].
    fn wait<T>(&mut self, mut step: impl FnMut(&mut Self) -> Result<Option<T>>) -> Result<T> {
        let deadline = Instant::now() + PATIENCE;
        loop {
            if let Some(value) = step(self)? {
                return Ok(value);
            }
            if Instant::now() >= deadline {
                return Err(Error::Device(format!(
                    "the device left a request unanswered for {}s",
                    PATIENCE.as_secs()
                )));
            }
            std::hint::spin_loop();
        }
    }

    pub(crate) fn idle(&self) -> bool {
        self.answers.iter().all(|answer| answer.id.is_none())
    }

    /// Writes an unpublished frame when both the queue and its answer slot have room.
    pub(crate) fn prepare(&mut self, request: &Request<'_>) -> Result<Option<Prepared<'_>>> {
        let id = self.next;
        let slot = id as usize % IN_FLIGHT;
        if self.answers[slot].id.is_some() {
            return Ok(None);
        }
        let mut frame = [0; 1 + MAX_SUBMISSION_WORDS];
        let words = wire::encode(request, &mut frame[1..])
            .map_err(|why| Error::Device(format!("encoding request {id}: {why}")))?;
        frame[0] = header::pack(id, words as u8);
        let frame = &frame[..1 + words];
        let Poll::Ready(publication) = self.queues.submission.prepare(frame) else {
            return Ok(None);
        };
        Ok(Some(Prepared {
            publication: publication.map_err(|why| Error::Device(format!("submitting request {id}: {why}")))?,
            answer: &mut self.answers[slot],
            next: &mut self.next,
        }))
    }

    /// The answer to `id` once it is complete. The device answers requests as they finish, in any
    /// order; answers to other requests wait in their slots for their own callers.
    pub(crate) fn poll(&mut self, id: u32) -> Result<Option<Response>> {
        self.collect()?;
        let answer = &mut self.answers[id as usize % IN_FLIGHT];
        if answer.id != Some(id) {
            return Ok(None);
        }
        let response = match answer.decode() {
            Ok(None) => return Ok(None),
            response => {
                response.map_err(|why| Error::Device(format!("the device's answer to {id} is unreadable: {why}")))
            }
        };
        answer.id = None;
        answer.words.clear();
        response
    }

    /// Takes every completion the device has published into the slot of the request it answers.
    fn collect(&mut self) -> Result<()> {
        while let Some(Completion { id, payload }) = self.receive()? {
            let answer = &mut self.answers[id as usize % IN_FLIGHT];
            if answer.id != Some(id) || !matches!(answer.decode(), Ok(None)) {
                return Err(Error::Device(format!("unexpected completion for request {id}")));
            }
            answer.words.extend_from_slice(&payload);
        }
        Ok(())
    }

    /// Submits `request` on an idle link, whose queue and answer slots are free, and spins for
    /// its answer.
    pub(crate) fn call(&mut self, request: &Request<'_>) -> Result<Response> {
        let id = self
            .prepare(request)?
            .ok_or_else(|| Error::Device("a call found its link busy".into()))?
            .publish();
        self.wait(|link| link.poll(id))
    }
}

/// Cluster `cluster`'s refusal of a `request`, with the firmware's status and detail.
pub(crate) fn failure(cluster: usize, request: &str, status: i32, message: &str) -> Error {
    if status == Status::Panicked as i32 {
        return Error::Panicked(message.into());
    }
    let status = Status::try_from(status).map_or_else(|code| format!("status {code}"), |status| status.to_string());
    let separator = if message.is_empty() { "" } else { ": " };
    Error::Device(format!(
        "rank {cluster} failed a {request}: {status}{separator}{message}"
    ))
}

/// A cluster's ARM counter in ticks, read between two host `CLOCK_MONOTONIC_RAW` readings in ns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub before: u64,
    pub counter: u64,
    pub after: u64,
}

impl Link {
    /// One counter read; `None` while the cluster has work in flight, which the read would wait
    /// behind.
    pub(crate) fn sample(&mut self) -> Result<Option<Sample>> {
        if !self.idle() {
            return Ok(None);
        }
        let before = now()?;
        let Response::Clock { counter } = self.call(&Request::Clock)? else {
            return Err(Error::Device("invalid clock response".into()));
        };
        let after = now()?;
        Ok(Some(Sample { before, counter, after }))
    }
}

fn now() -> Result<u64> {
    let mut time = nix::libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `time` is writable storage for the requested monotonic clock.
    if unsafe { nix::libc::clock_gettime(nix::libc::CLOCK_MONOTONIC_RAW, &mut time) } != 0 {
        return Err(Error::Device(std::io::Error::last_os_error().to_string()));
    }
    Ok(time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::FileExt;

    /// A task panic reaches the caller as `Panicked` with the task's message, so it can re-raise it.
    #[test]
    fn reports_task_panics_as_panics() {
        let error = failure(0, "launch", Status::Panicked as i32, "panicked at kernel.rs:3:5");
        assert!(matches!(error, Error::Panicked(message) if message == "panicked at kernel.rs:3:5"));
    }

    /// A cluster's submission credit returns only when its answer is consumed.
    #[test]
    fn retains_unconsumed_answers() {
        let (mut link, file) = crate::test::link();
        let request = Request::Launch {
            args: furiosa_opt_ipc::Args::Inline(&[]),
            executable: furiosa_opt_ipc::Staged {
                addr: 0,
                len: 1,
                token: 1,
            },
        };
        for _ in 0..IN_FLIGHT {
            link.prepare(&request).unwrap().unwrap().publish();
        }
        assert!(link.prepare(&request).unwrap().is_none());
        crate::test::answer(&file, 1);
        link.collect().unwrap();
        let (mut consumed, mut produced) = ([0; 4], [0; 4]);
        file.read_exact_at(&mut consumed, 4).unwrap();
        file.read_exact_at(&mut produced, 0).unwrap();
        assert_eq!(consumed, produced);
        assert!(link.prepare(&request).unwrap().is_none());
        assert!(matches!(link.poll(1).unwrap(), Some(Response::Launch)));
        assert!(link.prepare(&request).unwrap().is_some());
    }

    /// Answers arriving in any order reach the requests they answer.
    #[test]
    fn matches_answers_out_of_order() {
        let (mut link, file) = crate::test::link();
        let ids = [(); 3].map(|()| link.prepare(&Request::Clock).unwrap().unwrap().publish());
        let clock = |counter| Response::Clock { counter };
        crate::test::respond(&file, &[(ids[2], clock(7)), (ids[0], clock(5)), (ids[1], clock(6))]);
        for (id, counter) in ids.into_iter().zip([5, 6, 7]) {
            assert_eq!(link.poll(id).unwrap(), Some(Response::Clock { counter }));
        }
    }
}
