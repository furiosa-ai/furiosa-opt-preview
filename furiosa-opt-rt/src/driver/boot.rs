//! How a driver opens: firmware boot on every cluster, then each chip's PDMA configuration.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, OnceLock};
use std::task::Waker;

use furiosa_opt_abi::bootloader;
use furiosa_opt_ipc::{Request, Response};

use super::chip::Chip;
use super::cluster::Cluster;
use super::dma::DmaQueue;
use super::dram::Dram;
use super::link::{Link, PATIENCE};
use super::window::Window;
use super::{DRAM_BASE, Driver};
use crate::arch::rngd::{self, PeRange};
use crate::buffer::Ranges;
use crate::{Error, Placement, Result};

const FIRMWARE: &[u8] = include_bytes!(env!("FURIOSA_OPT_DEVICE_IMAGE"));
const _: () = assert!(FIRMWARE.len() > 8, "the firmware image opens with its magic word");
const BOOTLOADER: &[u8] = include_bytes!(env!("FURIOSA_OPT_BOOTLOADER"));

impl Driver {
    /// Reserves `placement` and boots every cluster on firmware staged through BAR4; the ranges
    /// are what the device's allocator may hand out. `worker` retires host transfers nobody
    /// observes.
    pub(crate) fn open(placement: Placement, worker: Waker) -> Result<(Self, Ranges)> {
        let (mut chips, shared_end, opened) = Self::reserve(&placement)?;
        let Placement { chips: members, pes } = placement;
        if let Some(deadline) = members.iter().filter_map(|&chip| rngd::timeout(chip)).min()
            && PATIENCE >= deadline
        {
            return Err(Error::Device(format!(
                "the driver's {deadline:?} deadline does not outlast the host's {PATIENCE:?} wait for a launch"
            )));
        }
        // Time and pid: unique among the stagings a boot could confuse, no more.
        let token = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64)
            ^ u64::from(std::process::id()) << 48;
        let clusters = opened.len() / members.len();
        // A chip's window is shared by its clusters, so its share is theirs together.
        let mask = pes.clone().fold(0, |mask, pe| mask | 1 << pe);
        let mut drams = members
            .iter()
            .map(|&chip| Dram::open(chip, mask, FIRMWARE.len()))
            .collect::<Result<Vec<_>>>()?;
        // The finest level this process logs the firmware at; its lines reach `pe_log` at and above it.
        let log_level = log::Level::iter()
            .filter(|&level| log::log_enabled!(target: "furiosa_opt_firmware", level))
            .last()
            .map_or(0, |level| level as u8);
        let mut links = opened
            .into_iter()
            .enumerate()
            .map(|(cluster, queues)| {
                let rank = cluster / clusters;
                let mut link = Link::new(queues);
                let window = Self::boot(&mut link, &mut drams[rank], &chips[rank].window, token)?;
                chips[rank].base = DRAM_BASE + window;
                match link.call(&Request::Initialize {
                    physical: chips[rank].physical,
                    rank: rank as u8,
                    chips: members.len() as u8,
                    cluster: cluster as u8,
                    log_level,
                })? {
                    Response::Initialize => Ok(link),
                    response => Err(Error::Device(format!(
                        "npu{} initialization: {response:?}",
                        members[rank]
                    ))),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        for (rank, (dram, chip)) in drams.iter().zip(&mut chips).enumerate() {
            let (peers, base) = (chip.peers, chip.base);
            let Some(pdma) = chip.open_dma(rank, members.len())? else {
                continue;
            };
            for column in 0..clusters {
                let link = &mut links[rank * clusters + column];
                let config = pdma.config(
                    base + dram.free().end as u64,
                    peers,
                    members.len() as u8,
                    column,
                    clusters,
                );
                link.queues.channels = pdma.channels(config.channels);
                match link.call(&Request::Dma(config))? {
                    Response::Dma => {}
                    response => return Err(Error::Device(format!("DMA configuration: {response:?}"))),
                }
            }
        }
        // An allocation is one offset on every chip.
        let free = drams
            .iter()
            .map(Dram::free)
            .fold(0..usize::MAX, |all, one| all.start.max(one.start)..all.end.min(one.end));
        let window = chips.iter().map(|chip| chip.base - DRAM_BASE).max().unwrap_or(0);
        let links = Arc::new(Mutex::new(links));
        let transfers = DmaQueue::open(members.iter().copied(), Arc::clone(&links), worker)?;
        let driver = Self {
            links,
            topology: (members.len() as u8, pes.len() as u8).into(),
            chips,
            transfers,
            failure: OnceLock::new(),
            tokens: AtomicU64::new(0),
            owners: AtomicU64::new(0),
        };
        let end = if members.len() == 1 {
            free.end
        } else {
            shared_end.saturating_sub(window as usize)
        };
        Ok((driver, Ranges::new(free, end)))
    }

    /// Reserves `placement`'s PEs on every one of its chips; PEs another process holds are
    /// [`Error::Reserved`]. Holding PEs is the only way to learn they are free.
    pub(crate) fn reserve(Placement { chips, pes }: &Placement) -> Result<(Vec<Chip>, usize, Vec<Cluster>)> {
        let ascending = chips.windows(2).all(|pair| pair[0] < pair[1]);
        let max = rngd::MAX_CHIPS;
        if !(1..=usize::from(max)).contains(&chips.len()) || !ascending || chips.len() > 1 && pes.len() < 8 {
            return Err(Error::Invalid(format!(
                "{chips:?} is not a device: 1 to {max} ascending chips, each with all PEs when several"
            )));
        }
        let ranges = PeRange::clusters(pes.clone())?;
        let (joined, shared_end) = Chip::join(chips)?;
        let clusters = std::thread::scope(|scope| {
            let opens = chips
                .iter()
                .enumerate()
                .flat_map(|(rank, &chip)| ranges.iter().map(move |&pes| (rank, chip, pes)))
                .map(|(rank, chip, pes)| scope.spawn(move || Cluster::open(chip, pes, rank as u8, BOOTLOADER)))
                .collect::<Vec<_>>();
            opens
                .into_iter()
                .map(|open| {
                    open.join()
                        .unwrap_or_else(|_| Err(Error::Device("cluster open panicked".into())))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok((joined, shared_end, clusters))
    }

    /// Has the bootloader on `link` start the firmware image staged in `dram`, returning where
    /// the cluster sees the chip's window. A chip's first cluster writes the image through the
    /// chip's BAR4 `window`, which shows the same bytes wherever the cluster sees them, then finds
    /// the window, trying each candidate until a boot lands on it.
    fn boot(link: &mut Link, dram: &mut Dram, window: &Window, token: u64) -> Result<u64> {
        if dram.at.is_none() {
            let bytes = [&bootloader::magic(token).to_le_bytes()[..], &FIRMWARE[8..]].concat();
            window.write(dram.offset as u64, &bytes)?;
        }
        for candidate in dram.candidates() {
            let address = DRAM_BASE + candidate + dram.offset as u64;
            match link.boot(address, FIRMWARE.len() as u64, token)? {
                bootloader::Code::Booted => {
                    log::debug!("npu{} booted, its window at {candidate:#x}", dram.chip);
                    dram.at = Some(candidate);
                    return Ok(candidate);
                }
                bootloader::Code::NoImage => {}
                other => {
                    return Err(Error::Device(format!(
                        "the bootloader refused the firmware image: {other:?}"
                    )));
                }
            }
        }
        Err(Error::Device(format!(
            "no candidate window showed the firmware image staged at {:#x}",
            dram.offset
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Device::new` refuses a placement that is not ascending chips, all PEs when several.
    #[test]
    fn rejects_placements_that_are_not_devices() {
        for (chips, pes) in [(vec![], 0..8), (vec![3, 2], 0..8), (vec![0, 1], 0..4)] {
            let open = Driver::open(Placement { chips, pes }, Waker::noop().clone());
            assert!(matches!(open, Err(Error::Invalid(_))));
        }
    }
}
