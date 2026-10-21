//! The layer that touches the kernel driver and the device's memory.

use std::os::fd::RawFd;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use arrayvec::ArrayVec;
use furiosa_opt_ipc::{Launch, LaunchOption, Profile, Records, Staged, Status};

use crate::arch::rngd;
use crate::buffer::Ranges;
use crate::function::{Executable, Operand, Prepared};
use crate::{Error, Result, Topology};

mod boot;
pub(crate) mod chip;
pub(crate) mod cluster;
pub(crate) mod dma;
pub(crate) mod dram;
pub(crate) mod kernel;
pub(crate) mod link;
pub(crate) mod mailbox;

use chip::{Chip, Pdma};
use mailbox::{Mailbox, PATIENCE, Sent};

/// The hardware one process reserved for a device.
pub(crate) struct Driver {
    /// One lock over every cluster's mailbox: a launch is admitted on all clusters or none, and its
    /// completion returns queue credit and SRAM together. Shared with the transfers, whose copies
    /// the clusters run too. Closed before the chips it runs on.
    pub(crate) mailboxes: Arc<Mutex<Vec<Mailbox>>>,
    /// Released after the clusters that run on it and before the chips it reserved.
    pub(crate) pdma: Option<Pdma>,
    /// In rank order: a view's chip index is an index here.
    pub(crate) chips: Vec<Chip>,
    pub(crate) transfers: dma::DmaQueue,
    /// Set once a launch or transfer fails; shared with the transfers.
    pub(crate) poison: Arc<Poison>,
    pub(crate) topology: Topology,
    /// The device memory the open left free, which the device's allocator hands out.
    pub(crate) allocatable: Ranges,
}

impl Driver {
    /// Submits a prepared launch to every cluster, each recording `profile`; its ticket bounds how
    /// long answers may take.
    pub(crate) fn submit(
        &self,
        executable: &Executable,
        prepared: &Prepared<'_>,
        profile: Option<Profile>,
    ) -> Result<Option<Ticket>> {
        let mut mailboxes = self.mailboxes.lock().unwrap();
        if let Some(error) = self.poison.get() {
            return Err(error.clone());
        }
        if !prepared.belongs_to(&executable.device) {
            return Err(Error::ForeignBuffer);
        }
        let mut args = [0; furiosa_opt_ipc::MAX_SUBMISSION_WORDS];
        let mut ticket = Ticket {
            ids: [None; MAX_CLUSTERS],
            replies: mailboxes.iter().map(|_| Records::new()).collect(),
            deadline: Instant::now() + PATIENCE,
            reservation: None,
        };
        let mut submissions = ArrayVec::<_, MAX_CLUSTERS>::new();
        for (cluster, mailbox) in mailboxes.iter_mut().enumerate() {
            let window = self.chip(cluster).base - DRAM_BASE;
            let args = prepared.encode(window, cluster, &mut args);
            let Some(submission) = mailbox
                .prepare(&Launch {
                    args,
                    executable: Staged {
                        addr: DRAM_BASE + window + executable.staging.addr() as u64,
                        len: executable.staging.len() as u64,
                        token: executable.token,
                    },
                    options: profile.map(LaunchOption::Profile).into_iter().collect(),
                })
                .map_err(|why| Error::Device(format!("rank {cluster}: {why}")))?
            else {
                return Ok(None);
            };
            submissions.push(submission);
        }
        ticket.reservation = Some(executable.device.allocations.reserve(executable.sram_end)?);
        // Send together so frame writes and queue waits do not separate cluster starts.
        for (id, submission) in ticket.ids.iter_mut().zip(submissions) {
            *id = Some(submission.send());
        }
        Ok(Some(ticket))
    }

    /// Advances `ticket`; once every cluster answered, releases its SRAM and returns what they
    /// replied. The caller retains possibly active memory after a failure.
    pub(crate) fn poll(&self, ticket: &mut Ticket) -> Result<Option<Replies>> {
        let mut mailboxes = self.mailboxes.lock().unwrap();
        if let Some(error) = self.poison.get() {
            return Err(error.clone());
        }
        let done = self.answers(&mut mailboxes, ticket).inspect_err(|error| {
            self.poison.set(error.clone());
        })?;
        Ok(done.then(|| {
            ticket.reservation = None;
            std::mem::take(&mut ticket.replies)
        }))
    }

    /// Writes every cluster's words for `operands`, one cluster after another.
    pub(crate) fn encode(&self, operands: &[Operand], words: &mut [u64]) {
        let clusters = self.topology.clusters();
        for (cluster, words) in words.chunks_mut(operands.len()).take(clusters).enumerate() {
            Operand::encode(operands, self.chip(cluster).base - DRAM_BASE, cluster, words);
        }
    }

    /// Each cluster's DMA segments for the dma-buf `fd`, in cluster order.
    pub(crate) fn register(&self, fd: RawFd) -> Result<Vec<Vec<rngd::dma::Segment>>> {
        let pdma = self.pdma()?;
        let mailboxes = self.mailboxes.lock().unwrap();
        (mailboxes.iter().enumerate())
            .map(|(cluster, mailbox)| {
                let (rank, column) = self.place(cluster);
                mailbox
                    .cluster()
                    .register(fd, pdma.channels(rank, column, self.columns()))
            })
            .collect()
    }

    /// Closes every cluster, which stops its task code.
    pub(crate) fn stop(&self) {
        self.mailboxes.lock().unwrap().clear();
    }

    /// The firmware's DMA, or [`Error::Unsupported`] without it.
    pub(crate) fn pdma(&self) -> Result<&Pdma> {
        self.pdma
            .as_ref()
            .ok_or_else(|| Error::Unsupported(NO_FIRMWARE_DMA.into()))
    }

    fn answers(&self, mailboxes: &mut [Mailbox], ticket: &mut Ticket) -> Result<bool> {
        let timed_out = Instant::now() >= ticket.deadline;
        for (cluster, mailbox) in mailboxes.iter_mut().enumerate() {
            let Some(sent) = ticket.ids[cluster] else { continue };
            let answer = mailbox
                .receive(sent)
                .map_err(|why| Error::Device(format!("rank {cluster}: {why}")))?;
            let Some(answer) = answer else {
                if timed_out {
                    return Err(Error::Device(format!(
                        "rank {cluster} left request {} unanswered for {}s",
                        sent.id(),
                        PATIENCE.as_secs()
                    )));
                }
                continue;
            };
            ticket.ids[cluster] = None;
            match answer {
                Ok(records) => ticket.replies[cluster] = records,
                // Firmware without PDMA channels fails the DMA a task issues with this bare status.
                Err(failure) if failure.status == Status::Internal as i32 && self.pdma.is_none() => {
                    return Err(Error::Unsupported(NO_FIRMWARE_DMA.into()));
                }
                Err(failure) => return Err(mailbox::refused::<Launch<'_>>(cluster, &failure)),
            }
        }
        Ok(ticket.ids.iter().all(Option::is_none))
    }

    /// The chip cluster `cluster` runs on.
    fn chip(&self, cluster: usize) -> &Chip {
        &self.chips[self.place(cluster).0]
    }

    /// The rank of the chip cluster `cluster` runs on, and its column there.
    fn place(&self, cluster: usize) -> (usize, usize) {
        (cluster / self.columns(), cluster % self.columns())
    }

    /// Clusters per chip.
    fn columns(&self) -> usize {
        Topology {
            chips: 1,
            ..self.topology
        }
        .clusters()
    }
}

/// One submitted launch: the request id every cluster answers it under, what the clusters that
/// answered replied, and the SRAM its temporaries hold until every cluster answered.
pub(crate) struct Ticket {
    ids: [Option<Sent<Records>>; MAX_CLUSTERS],
    replies: Replies,
    deadline: Instant,
    reservation: Option<crate::buffer::Reservation>,
}

/// Every cluster's reply to one launch, in cluster order.
pub(crate) type Replies = ArrayVec<Records, MAX_CLUSTERS>;

/// The first failure that left the device unusable. A launch runs on every cluster together, so
/// once one fails the clusters' state and memory are unknown, and nothing short of reopening the
/// device recovers it: every later launch and transfer is refused with this error.
#[derive(Default)]
pub(crate) struct Poison(OnceLock<Error>);

impl Poison {
    pub(crate) fn get(&self) -> Option<&Error> {
        self.0.get()
    }

    /// Records `error` unless an earlier failure poisoned the device first; the one recorded.
    pub(crate) fn set(&self, error: Error) -> &Error {
        self.0.get_or_init(|| error)
    }
}

const DRAM_BASE: u64 = rngd::DEVICE_MEMORY.start;

/// Why mapping pinned memory, host operands and device code's DMA are refused when the kernel
/// driver reserves no PDMA channels for the firmware.
pub(crate) const NO_FIRMWARE_DMA: &str = "the kernel driver reserves no PDMA channels for the firmware (no NPU_PDMA_CAPS_RESERVE), \
     so device code cannot move host or peer memory";

/// Clusters one device has at most: every cluster of every chip.
const MAX_CLUSTERS: usize = Topology {
    chips: rngd::MAX_CHIPS,
    pes: rngd::PES,
}
.clusters();

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::function::Inline;
    use crate::memory::Kind;
    use furiosa_opt_ipc::Clock;

    use super::{Executable, Prepared};
    use crate::test::{chip, driver, mailbox};

    /// A launch is admitted on every cluster or on none.
    #[test]
    fn admission_commits_all_clusters_together() {
        let device = crate::test::device(&[0], 8);
        let executable = Executable {
            staging: Arc::new(device.alloc(Kind::Dram, 512).unwrap()),
            device: Arc::new(device),
            token: 9,
            sram_end: crate::arch::rngd::sram::WINDOW.end as usize,
        };
        let (first, _) = mailbox();
        let (mut second, file) = mailbox();
        let clock = second.prepare(&Clock).unwrap().unwrap().send();
        for _ in 1..crate::driver::mailbox::IN_FLIGHT {
            second.prepare(&Clock).unwrap().unwrap().send();
        }
        let driver = driver(vec![chip(0, 0)], 8, vec![first, second]);
        let prepared = Prepared::Inline(Inline::new(&executable.device, &[]).unwrap());
        let idle = || driver.mailboxes.lock().unwrap()[0].idle();
        assert!(driver.submit(&executable, &prepared, None).unwrap().is_none());
        assert!(idle());

        crate::test::respond(&file, &[(1, furiosa_opt_ipc::Answer(Ok(1_u64)))]);
        assert_eq!(driver.mailboxes.lock().unwrap()[1].receive(clock).unwrap(), Some(Ok(1)));
        let occupied = executable.device.alloc(Kind::Sram, 256).unwrap();
        assert!(driver.submit(&executable, &prepared, None).is_err());
        assert!(idle());
        drop(occupied);
        crate::test::without_allocations(|| driver.submit(&executable, &prepared, None).unwrap().unwrap());
    }
}
