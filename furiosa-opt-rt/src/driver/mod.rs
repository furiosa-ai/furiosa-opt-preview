//! The layer that touches the kernel driver and the device's memory.

// allow: SIZE_OK — All-cluster publication and its ring fixtures share one admission contract.

use std::os::fd::RawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use arrayvec::ArrayVec;
use furiosa_opt_ipc::{Request, Response, Staged, Status};

use crate::arch::rngd;
use crate::function::{Executable, Operand, Prepared};
use crate::{Error, Result, Topology};

mod boot;
pub(crate) mod chip;
pub(crate) mod cluster;
pub(crate) mod dma;
pub(crate) mod dram;
pub(crate) mod link;
pub(crate) mod map;
pub(crate) mod window;

use chip::Chip;
use link::{Link, PATIENCE, Sample};

const DRAM_BASE: u64 = rngd::DEVICE_MEMORY.start;

/// Why device code cannot move host or peer memory on a device without firmware DMA.
pub(crate) const NO_FIRMWARE_DMA: &str = "the kernel driver cannot reserve PDMA channels for the firmware \
                                          (no NPU_PDMA_CAPS_RESERVE), so device code cannot move host or peer memory";

/// Clusters one device has at most: every cluster of every chip.
const MAX_CLUSTERS: usize = rngd::clusters(Topology {
    chips: rngd::MAX_CHIPS,
    pes: rngd::PES,
});

/// The hardware one process reserved for a device.
pub(crate) struct Driver {
    /// One lock over every cluster's link: a launch is admitted on all clusters or none, and its
    /// completion returns queue credit and SRAM together. Shared with the transfers, whose copies
    /// the clusters run too. Closed before the chips it runs on.
    pub(crate) links: Arc<Mutex<Vec<Link>>>,
    /// In rank order: a view's chip index is an index here.
    pub(crate) chips: Vec<Chip>,
    pub(crate) transfers: dma::DmaQueue,
    /// The first launch failure, which rejects every later launch.
    pub(crate) failure: OnceLock<Error>,
    pub(crate) topology: Topology,
    /// Numbers each load, so a cluster can tell a new image at a reused address from the one it
    /// already holds.
    pub(crate) tokens: AtomicU64,
    pub(crate) owners: AtomicU64,
}

/// One submitted launch: the request id every cluster answers it under, and the SRAM its
/// temporaries hold until every cluster answered.
pub(crate) struct Ticket {
    ids: [Option<u32>; MAX_CLUSTERS],
    deadline: Instant,
    reservation: Option<crate::buffer::Reservation>,
}

impl Driver {
    pub(crate) fn owner(&self) -> Result<u64> {
        self.owners
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |owner| owner.checked_add(1))
            .map_err(|_| Error::Capacity)
    }

    pub(crate) fn token(&self) -> Result<u64> {
        self.tokens
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |token| token.checked_add(1))
            .map_err(|_| Error::Capacity)
    }

    /// Whether device code moves data itself, on PDMA channels the driver reserved on every chip.
    pub(crate) fn firmware_dma(&self) -> bool {
        self.chips.iter().all(|chip| chip.dma.is_some())
    }

    /// The chip cluster `cluster` runs on.
    fn chip(&self, cluster: usize) -> &Chip {
        &self.chips[cluster
            / rngd::clusters(Topology {
                chips: 1,
                ..self.topology
            })]
    }

    /// Writes every cluster's words for `operands`, one cluster after another.
    pub(crate) fn encode(&self, operands: &[Operand], words: &mut [u64]) {
        let clusters = rngd::clusters(self.topology);
        for (cluster, words) in words.chunks_mut(operands.len()).take(clusters).enumerate() {
            Operand::encode(operands, self.chip(cluster).base - DRAM_BASE, cluster, words);
        }
    }

    /// Submits a prepared launch to every cluster; its ticket bounds how long answers may take.
    pub(crate) fn submit(&self, executable: &Executable, prepared: &Prepared<'_>) -> Result<Option<Ticket>> {
        let mut links = self.links.lock().unwrap();
        if let Some(error) = self.failure.get() {
            return Err(error.clone());
        }
        if !prepared.belongs_to(&executable.device) {
            return Err(Error::ForeignBuffer);
        }
        let mut args = [0; furiosa_opt_ipc::MAX_SUBMISSION_WORDS];
        let mut ticket = Ticket {
            ids: [None; MAX_CLUSTERS],
            deadline: Instant::now() + PATIENCE,
            reservation: None,
        };
        let mut submissions = ArrayVec::<_, MAX_CLUSTERS>::new();
        for (cluster, link) in links.iter_mut().enumerate() {
            let window = self.chip(cluster).base - DRAM_BASE;
            let args = prepared.encode(window, cluster, &mut args);
            let Some(submission) = link
                .prepare(&Request::Launch {
                    args,
                    executable: Staged {
                        addr: DRAM_BASE + window + executable.staging.addr() as u64,
                        len: executable.staging.len() as u64,
                        token: executable.token,
                    },
                })
                .map_err(|why| Error::Device(format!("rank {cluster}: {why}")))?
            else {
                return Ok(None);
            };
            submissions.push(submission);
        }
        ticket.reservation = Some(executable.device.allocations.reserve(executable.sram_end)?);
        // Publish indices together so frame writes and queue waits do not separate cluster starts.
        for (id, submission) in ticket.ids.iter_mut().zip(submissions) {
            *id = Some(submission.publish());
        }
        Ok(Some(ticket))
    }

    /// Advances `ticket`, releasing its SRAM once every cluster answered; the caller retains
    /// possibly active memory after a failure.
    pub(crate) fn poll(&self, ticket: &mut Ticket) -> Result<bool> {
        let mut links = self.links.lock().unwrap();
        if let Some(error) = self.failure.get() {
            return Err(error.clone());
        }
        let done = self.answers(&mut links, ticket).inspect_err(|error| {
            self.failure.get_or_init(|| error.clone());
        })?;
        if done {
            ticket.reservation = None;
        }
        Ok(done)
    }

    fn answers(&self, links: &mut [Link], ticket: &mut Ticket) -> Result<bool> {
        let timed_out = Instant::now() >= ticket.deadline;
        for (cluster, link) in links.iter_mut().enumerate() {
            let Some(id) = ticket.ids[cluster] else { continue };
            let answer = link
                .poll(id)
                .map_err(|why| Error::Device(format!("rank {cluster}: {why}")))?;
            let Some(answer) = answer else {
                if timed_out {
                    return Err(Error::Device(format!(
                        "rank {cluster} left request {id} unanswered for {}s",
                        PATIENCE.as_secs()
                    )));
                }
                continue;
            };
            ticket.ids[cluster] = None;
            match answer {
                Response::Launch => continue,
                // Firmware without PDMA channels fails the DMA a task issues with this bare status.
                Response::Error { status, message } => {
                    let message = if status == Status::Internal as i32 && !self.firmware_dma() {
                        NO_FIRMWARE_DMA
                    } else {
                        &message
                    };
                    return Err(link::failure(cluster, "launch", status, message));
                }
                other => {
                    return Err(Error::Device(format!(
                        "rank {cluster} answered a launch with {other:?}"
                    )));
                }
            }
        }
        Ok(ticket.ids.iter().all(Option::is_none))
    }

    /// One counter read of cluster `cluster`; `None` while it has work in flight.
    pub(crate) fn sample(&self, cluster: usize) -> Result<Option<Sample>> {
        self.links
            .lock()
            .unwrap()
            .get_mut(cluster)
            .ok_or_else(|| Error::Invalid(format!("the device has no cluster {cluster}")))?
            .sample()
    }

    /// Each cluster's DMA segments for the dma-buf `fd`, in cluster order.
    pub(crate) fn register(&self, fd: RawFd) -> Result<Vec<Vec<rngd::dma::Segment>>> {
        let links = self.links.lock().unwrap();
        if links.is_empty() {
            return Err(Error::Device("firmware DMA is unavailable".into()));
        }
        links.iter().map(|link| link.queues.map(fd)).collect()
    }

    /// Closes every cluster, which stops its task code.
    pub(crate) fn stop(&self) {
        self.links.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::FileExt;
    use std::sync::Arc;

    use crate::function::{Indirect, Inline};
    use crate::memory::Kind;
    use furiosa_opt_abi::ring::Completion;
    use furiosa_opt_ipc::{Args, Request, Response, Staged, wire};

    use super::{DRAM_BASE, Executable, Operand, Prepared};
    use crate::test::{chip, driver, link};

    fn executable(device: crate::Device, sram_end: usize) -> Executable {
        Executable {
            staging: Arc::new(device.alloc(Kind::Dram, 512).unwrap()),
            device: Arc::new(device),
            token: 9,
            sram_end,
        }
    }

    /// Each cluster's launch frame carries its chip's operand addresses, inline or by table.
    #[test]
    fn binds_cluster_arguments() {
        let device = crate::test::device(&[0, 1], 8);
        let mut buffer = device.alloc(Kind::Dram, 8).unwrap();
        let address = buffer.addr() as u64;
        let mut host = crate::test::mapped(&device, vec![0, 256, 512, 768], 4096);
        let host_table = host.mapping.as_ref().unwrap().table.addr() as u64;
        let inline = [
            crate::Argument::Value(71),
            buffer.slice_mut(..).unwrap().into(),
            crate::Argument::Host(host.slice_mut(16..)),
        ]
        .map(|argument| Operand::new(argument, &device, crate::buffer::Owner::Unordered).unwrap());
        let values = (0..128).map(Operand::Value).collect::<Vec<_>>();
        let indirect = device.alloc(Kind::Dram, values.len() * 2 * 8).unwrap();
        let executable = executable(device, 0);
        for (arguments, table) in [(&inline[..], None), (&values[..], Some(&indirect))] {
            let (links, files): (Vec<_>, Vec<_>) = (0..4).map(|_| link()).unzip();
            let driver = driver(vec![chip(0, 65536), chip(1, 131072)], 8, links);
            let prepared = match table {
                None => Prepared::Inline(Inline::new(&executable.device, arguments).unwrap()),
                Some(table) => Prepared::Indirect(Indirect::new(&executable.device, table, arguments.len()).unwrap()),
            };
            crate::test::without_allocations(|| {
                assert!(driver.submit(&executable, &prepared).unwrap().is_some());
            });
            let mut encoded = [0; 12];
            driver.encode(&inline, &mut encoded);
            for (column, file) in files.iter().enumerate() {
                let window = ((column / 2 + 1) * 65536) as u64;
                let host =
                    crate::arch::rngd::dma::Operand::new(DRAM_BASE + host_table + window + column as u64 * 256, 16);
                let words = [71, address + window, host.unwrap().encode()];
                assert_eq!(encoded[column * 3..][..3], words);
                let mut frame = [0_u64; 8];
                file.read_exact_at(bytemuck::cast_slice_mut(&mut frame), 4096 + 8)
                    .unwrap();
                assert_eq!(
                    wire::decode::<Request<'_>>(&frame).unwrap(),
                    Request::Launch {
                        args: match table {
                            None => Args::Inline(&words),
                            Some(_) => Args::Indirect {
                                address: DRAM_BASE + window + (indirect.addr() + column % 2 * values.len() * 8) as u64,
                                count: values.len() as u32,
                            },
                        },
                        executable: Staged {
                            addr: DRAM_BASE + window + executable.staging.addr() as u64,
                            len: 512,
                            token: 9,
                        },
                    }
                );
            }
        }
    }

    /// A launch is admitted on every cluster or on none.
    #[test]
    fn admission_commits_all_clusters_together() {
        let executable = executable(
            crate::test::device(&[0], 8),
            crate::arch::rngd::sram::WINDOW.end as usize,
        );
        let (first, _) = link();
        let (mut second, file) = link();
        for _ in 0..crate::driver::link::IN_FLIGHT {
            second.prepare(&Request::Clock).unwrap().unwrap().publish();
        }
        let driver = driver(vec![chip(0, 0)], 8, vec![first, second]);
        let prepared = Prepared::Inline(Inline::new(&executable.device, &[]).unwrap());
        let room = || driver.links.lock().unwrap()[0].queues.submission.space().unwrap();
        let idle = || driver.links.lock().unwrap()[0].idle();
        let before = room();
        assert!(driver.submit(&executable, &prepared).unwrap().is_none());
        assert!(idle());
        assert_eq!(room(), before);

        let mut response = [0; 3];
        let words = wire::encode(&Response::Clock { counter: 1 }, &mut response).unwrap();
        let frame = Completion::encode(1, &response[..words]).next().unwrap();
        file.write_all_at(bytemuck::cast_slice(&frame), 8).unwrap();
        file.write_all_at(&(frame.len() as u32).to_le_bytes(), 0).unwrap();
        driver.links.lock().unwrap()[1].poll(1).unwrap().unwrap();
        let occupied = executable.device.alloc(Kind::Sram, 256).unwrap();
        assert!(driver.submit(&executable, &prepared).is_err());
        assert!(idle());
        assert_eq!(room(), before);
        drop(occupied);
        crate::test::without_allocations(|| driver.submit(&executable, &prepared).unwrap().unwrap());
    }
}
