//! The host side of one cluster's firmware: requests out on the cluster's link, answers back on
//! it, matched by request id.

use super::cluster::Cluster;
use super::kernel::map::Mapping;
use super::link::Link;
use crate::{Error, Result};
use furiosa_opt_abi::bootloader;
use furiosa_opt_abi::ring::{Completion, header};
use furiosa_opt_ipc::{Answer, Call, Failure, LAUNCH_TIMEOUT, MAX_SUBMISSION_WORDS, Payload, Staged, Status, frame};
use std::marker::PhantomData;
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

/// One cluster's link, with the requests in flight on it, and the cluster it reaches.
pub(crate) struct Mailbox {
    link: Link,
    /// Dropped after the link: closing it stops the cluster's PEs.
    cluster: Cluster,
    next: u32,
    answers: [Slot; IN_FLIGHT],
}

impl Mailbox {
    /// The mailbox of `cluster`, whose firmware runs and answers over `link`.
    pub(crate) fn new(cluster: Cluster, link: Link) -> Self {
        Self {
            link,
            cluster,
            next: 1,
            answers: Default::default(),
        }
    }

    /// Writes an unsent frame when both the queue and its answer slot have room.
    pub(crate) fn prepare<C: Call>(&mut self, call: &C) -> Result<Option<Prepared<'_, C::Reply>>> {
        let id = self.next;
        let slot = id as usize % IN_FLIGHT;
        if self.answers[slot].id.is_some() {
            return Ok(None);
        }
        let mut frame = [0; 1 + MAX_SUBMISSION_WORDS];
        let words = frame::encode(&call.request(), &mut frame[1..])
            .map_err(|why| Error::Device(format!("encoding request {id}: {why}")))?;
        frame[0] = header::pack(id, words as u8);
        let frame = &frame[..1 + words];
        let Poll::Ready(publication) = self.link.submission().prepare(frame) else {
            return Ok(None);
        };
        Ok(Some(Prepared {
            publication: publication.map_err(|why| Error::Device(format!("submitting request {id}: {why}")))?,
            slot: &mut self.answers[slot],
            next: &mut self.next,
            reply: PhantomData,
        }))
    }

    /// The answer to `sent` once it is complete. The device answers requests as they finish, in
    /// any order; answers to other requests wait in their slots for their own callers.
    pub(crate) fn receive<R: Payload>(&mut self, sent: Sent<R>) -> Result<Option<std::result::Result<R, Failure>>> {
        let id = sent.id;
        self.collect()?;
        let slot = &mut self.answers[id as usize % IN_FLIGHT];
        if slot.id != Some(id) || !frame::Header::arrived(&slot.words) {
            return Ok(None);
        }
        let answer = frame::decode::<Answer<R>>(&slot.words)
            .map(|Answer(answer)| answer)
            .map_err(|why| Error::Device(format!("the device's answer to {id} is unreadable: {why}")));
        slot.id = None;
        slot.words.clear();
        answer.map(Some)
    }

    /// Submits `call` on an idle mailbox, whose queue and answer slots are free, and spins for
    /// its answer.
    pub(crate) fn call<C: Call>(&mut self, call: &C) -> Result<std::result::Result<C::Reply, Failure>> {
        let sent = self
            .prepare(call)?
            .ok_or_else(|| Error::Device("a call found its mailbox busy".into()))?
            .send();
        let deadline = Instant::now() + PATIENCE;
        loop {
            if let Some(answer) = self.receive(sent)? {
                return Ok(answer);
            }
            if Instant::now() >= deadline {
                return Err(Error::Device(format!(
                    "the device left request {} unanswered for {}s",
                    sent.id(),
                    PATIENCE.as_secs()
                )));
            }
            std::hint::spin_loop();
        }
    }

    pub(crate) fn idle(&self) -> bool {
        self.answers.iter().all(|answer| answer.id.is_none())
    }

    /// The cluster this mailbox talks to.
    pub(crate) fn cluster(&self) -> &Cluster {
        &self.cluster
    }

    /// Takes every completion the device has published into the slot of the request it answers.
    fn collect(&mut self) -> Result<()> {
        while let Some(Completion { id, payload }) = self.link.receive()? {
            let slot = &mut self.answers[id as usize % IN_FLIGHT];
            if slot.id != Some(id) || (!slot.words.is_empty() && frame::Header::arrived(&slot.words)) {
                return Err(Error::Device(format!("unexpected completion for request {id}")));
            }
            slot.words.extend_from_slice(&payload);
        }
        Ok(())
    }
}

pub(crate) struct Prepared<'a, R> {
    publication: furiosa_opt_abi::ring::Publication<'a, Arc<Mapping>>,
    slot: &'a mut Slot,
    next: &'a mut u32,
    reply: PhantomData<fn() -> R>,
}

impl<R> Prepared<'_, R> {
    pub(crate) fn send(self) -> Sent<R> {
        let id = *self.next;
        self.slot.id = Some(id);
        *self.next = id.wrapping_add(1);
        self.publication.publish();
        Sent { id, reply: PhantomData }
    }
}

/// A sent request, which only its own reply type answers.
pub(crate) struct Sent<R> {
    id: u32,
    reply: PhantomData<fn() -> R>,
}

impl<R> Clone for Sent<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for Sent<R> {}

impl<R> Sent<R> {
    /// The request id the firmware answers under.
    pub(crate) fn id(self) -> u32 {
        self.id
    }
}

/// Reply storage grows for diagnostics; ordinary completions reuse its capacity.
#[derive(Default)]
struct Slot {
    id: Option<u32>,
    words: Vec<u64>,
}

/// A cluster whose PEs run the bootloader, before any firmware does.
pub(crate) struct Bootloader {
    link: Link,
    cluster: Cluster,
    next: u32,
}

impl Bootloader {
    pub(crate) fn new(cluster: Cluster, link: Link) -> Self {
        Self { link, cluster, next: 1 }
    }

    /// Has the bootloader verify and start the firmware image `image`.
    pub(crate) fn boot(mut self, image: Staged) -> Result<Boot> {
        let id = self.next;
        self.next = id.wrapping_add(1);
        let request = bootloader::Request {
            addr: image.addr,
            len: image.len,
            token: image.token,
        };
        let Poll::Ready(result) = self.link.submission().write_all(&request.encode(id)) else {
            return Err(Error::Device("the boot submission queue is full".into()));
        };
        result.map_err(|why| Error::Device(format!("submitting the boot request: {why}")))?;
        let deadline = Instant::now() + PATIENCE;
        let completion = loop {
            if let Some(completion) = self.link.receive()? {
                break completion;
            }
            if Instant::now() >= deadline {
                return Err(Error::Device(format!(
                    "the bootloader left its request unanswered for {}s",
                    PATIENCE.as_secs()
                )));
            }
            std::hint::spin_loop();
        };
        let (answered, code) = (completion.id, completion.payload[0]);
        if answered != id {
            return Err(Error::Device(format!(
                "boot completion {answered} does not answer {id}"
            )));
        }
        match bootloader::Code::try_from(code) {
            Ok(bootloader::Code::Booted) => Ok(Boot::Running(Box::new(Mailbox::new(self.cluster, self.link)))),
            Ok(bootloader::Code::NoImage) => Ok(Boot::NoImage(self)),
            Ok(other) => Err(Error::Device(format!(
                "the bootloader refused the firmware image: {other:?}"
            ))),
            Err(code) => Err(Error::Device(format!("the bootloader answered with {code}"))),
        }
    }
}

/// What a boot left the cluster running.
pub(crate) enum Boot {
    /// The firmware, which now serves requests.
    Running(Box<Mailbox>),
    /// Still the bootloader: it found no image where the boot pointed.
    NoImage(Bootloader),
}

/// Cluster `cluster`'s refusal of request `C`, with the firmware's status and detail.
pub(crate) fn refused<C: Call>(cluster: usize, Failure { status, message }: &Failure) -> Error {
    let (request, status, message) = (C::NAME, *status, message.as_str());
    if status == Status::Panicked as i32 {
        return Error::Panicked(message.into());
    }
    let status = Status::try_from(status).map_or_else(|code| format!("status {code}"), |status| status.to_string());
    let separator = if message.is_empty() { "" } else { ": " };
    Error::Device(format!(
        "rank {cluster} failed a {request}: {status}{separator}{message}"
    ))
}

/// How long the host waits for an answer: past [`LAUNCH_TIMEOUT`], so a launch the device timed out still
/// answers first, and short of the driver's own doorbell deadline, 60 s by default.
pub(crate) const PATIENCE: Duration = LAUNCH_TIMEOUT.saturating_add(Duration::from_secs(2));

/// Requests one mailbox keeps in flight at most; a submitter past this many waits for an answer
/// to be collected.
pub(crate) const IN_FLIGHT: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_ipc::Clock;

    /// A task panic reaches the caller as `Panicked` with the task's message, so it can re-raise it.
    #[test]
    fn reports_task_panics_as_panics() {
        let failure = Failure::new(Status::Panicked, "panicked at kernel.rs:3:5");
        let error = refused::<furiosa_opt_ipc::Launch<'_>>(0, &failure);
        assert!(matches!(error, Error::Panicked(message) if message == "panicked at kernel.rs:3:5"));
    }

    /// Answers arriving in any order reach the requests they answer.
    #[test]
    fn matches_answers_out_of_order() {
        let (mut mailbox, file) = crate::test::mailbox();
        let sent = [(); 3].map(|()| mailbox.prepare(&Clock).unwrap().unwrap().send());
        let answers = [
            (sent[2].id, Answer(Ok(7_u64))),
            (sent[0].id, Answer(Ok(5))),
            (sent[1].id, Answer(Ok(6))),
        ];
        crate::test::respond(&file, &answers);
        for (sent, counter) in sent.into_iter().zip([5, 6, 7]) {
            assert_eq!(mailbox.receive(sent).unwrap(), Some(Ok(counter)));
        }
    }
}
