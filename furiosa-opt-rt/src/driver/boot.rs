//! How a driver opens: firmware boot on every cluster, then each chip's PDMA configuration.

use std::sync::{Arc, Mutex};
use std::task::Waker;
use std::time::Duration;

use furiosa_opt_abi::bootloader;
use furiosa_opt_ipc::{Configure, ConfigureOption, DRIVER_DEADLINE, Identity, Staged, launch_timeout};

use super::chip::{Group, Pdma};
use super::cluster::{Cluster, PeRange};
use super::dma::DmaQueue;
use super::dram::Dram;
use super::kernel::{self, window::Window};
use super::mailbox::{self, Boot, Bootloader, Mailbox};
use super::{DRAM_BASE, Driver};
use crate::arch::rngd;
use crate::buffer::Ranges;
use crate::{Error, Placement, Result};

impl Driver {
    /// Reserves `placement`, boots every cluster on firmware staged through BAR4 and hands each
    /// its chip's PDMA channels. `worker` is woken to collect host transfers nobody observes.
    pub(crate) fn open(placement: Placement, worker: Waker) -> Result<Self> {
        // The host waits as long as the driver does; the device fails a launch a second sooner, so
        // its answer naming the timeout comes first.
        let patience = (placement.chips.iter())
            .map(|&chip| kernel::timeout(chip).unwrap_or(DRIVER_DEADLINE))
            .min()
            .unwrap_or(DRIVER_DEADLINE);
        let timeout = launch_timeout(patience);
        if timeout.is_zero() {
            return Err(Error::Device(format!(
                "the driver's {patience:?} deadline leaves a launch no time"
            )));
        }
        let Reserved {
            group: Group {
                chips: joined,
                shared_end,
            },
            clusters: reserved,
        } = Self::reserve(&placement, patience)?;
        let Placement { chips: members, pes } = placement;
        // Time and pid: unique among the stagings a boot could confuse, no more.
        let token = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64)
            ^ u64::from(std::process::id()) << 48;
        let clusters = reserved.len() / members.len();
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
        let mut mailboxes = reserved
            .into_iter()
            .enumerate()
            .map(|(index, loader)| {
                let rank = index / clusters;
                Self::boot(loader, &mut drams[rank], &joined[rank].window, token)
            })
            .collect::<Result<Vec<_>>>()?;
        let physical = joined.iter().map(|chip| chip.physical).collect::<Vec<_>>();
        let chips = (joined.into_iter())
            .map(|chip| chip.open(DRAM_BASE))
            .collect::<Vec<_>>();
        let pdma = Pdma::reserve(&chips)?;
        for (index, mailbox) in mailboxes.iter_mut().enumerate() {
            let (rank, column) = (index / clusters, index % clusters);
            let (chip, dram) = (&chips[rank], &drams[rank]);
            let configure = Configure {
                options: [
                    Some(ConfigureOption::Identity(Identity {
                        physical: physical[rank],
                        rank: rank as u8,
                        chips: members.len() as u8,
                        cluster: index as u8,
                    })),
                    Some(ConfigureOption::LogLevel(log_level)),
                    Some(ConfigureOption::LaunchTimeout(
                        u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX),
                    )),
                    (pdma.as_ref()).map(|pdma| {
                        let scratch = chip.base + dram.free.end as u64;
                        ConfigureOption::Dma(pdma.config(chip, rank, scratch, column, clusters))
                    }),
                ]
                .into_iter()
                .flatten()
                .collect(),
            };
            mailbox
                .call(&configure)?
                .map_err(|failure| mailbox::refused::<Configure>(index, &failure))?;
        }
        // An allocation is one offset on every chip.
        let free = drams
            .iter()
            .map(|dram| dram.free.clone())
            .fold(0..usize::MAX, |all, one| all.start.max(one.start)..all.end.min(one.end));
        let window = chips.iter().map(|chip| chip.base - DRAM_BASE).max().unwrap_or(0);
        let end = if members.len() == 1 {
            free.end
        } else {
            // Counted from the device's share of DRAM.
            shared_end.saturating_sub(window as usize)
        };
        let mailboxes = Arc::new(Mutex::new(mailboxes));
        let dma = DmaQueue::open(members.iter().copied(), Arc::clone(&mailboxes), worker)?;
        Ok(Self {
            poison: Arc::clone(dma.poison()),
            dma,
            mailboxes,
            pdma,
            topology: (members.len() as u8, pes.len() as u8).into(),
            chips,
            allocatable: Ranges::new(free, end),
            patience,
        })
    }

    /// Reserves `placement`'s PEs on every one of its chips; PEs another process holds are
    /// [`Error::Reserved`]. Holding PEs is the only way to learn they are free.
    /// Each cluster's bootloader and firmware wait `patience` for an answer.
    pub(crate) fn reserve(Placement { chips, pes }: &Placement, patience: Duration) -> Result<Reserved> {
        let ascending = chips.windows(2).all(|pair| pair[0] < pair[1]);
        let max = rngd::MAX_CHIPS;
        if !(1..=usize::from(max)).contains(&chips.len()) || !ascending || chips.len() > 1 && pes.len() < 8 {
            return Err(Error::Invalid(format!(
                "{chips:?} is not a device: 1 to {max} ascending chips, each with all PEs when several"
            )));
        }
        let ranges = PeRange::clusters(pes.clone())?;
        let group = Group::join(chips)?;
        let clusters = std::thread::scope(|scope| {
            let opens = chips
                .iter()
                .enumerate()
                .flat_map(|(rank, &chip)| ranges.iter().map(move |&pes| (rank, chip, pes)))
                .map(|(rank, chip, pes)| {
                    scope.spawn(move || {
                        let (cluster, link) = Cluster::open(chip, pes, rank as u8, BOOTLOADER)?;
                        Ok(Bootloader::new(cluster, link, patience))
                    })
                })
                .collect::<Vec<_>>();
            opens
                .into_iter()
                .map(|open| {
                    open.join()
                        .unwrap_or_else(|_| Err(Error::Device("cluster open panicked".into())))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(Reserved { group, clusters })
    }

    /// Has the bootloader `loader` start the firmware image staged in `dram`. A chip's first
    /// cluster writes the image through the chip's BAR4 `window`; the cluster sees that byte at the
    /// same offset past its memory's start, as [`rngd::DRAM_FIRMWARE`] lays out.
    fn boot(loader: Bootloader, dram: &mut Dram, window: &Window, token: u64) -> Result<Mailbox> {
        if !dram.staged {
            let bytes = [&bootloader::magic(token).to_le_bytes()[..], &FIRMWARE[8..]].concat();
            window.write(dram.offset as u64, &bytes)?;
            dram.staged = true;
        }
        let addr = DRAM_BASE + dram.offset as u64;
        let image = Staged {
            addr,
            len: FIRMWARE.len() as u64,
            token,
        };
        match loader.boot(image)? {
            Boot::Running(mailbox) => Ok(*mailbox),
            Boot::NoImage => Err(Error::Device(format!(
                "npu{}: the bootloader found no firmware image at {addr:#x}, where the host wrote it: \
                 the kernel driver lays memory out unlike the PF build this runtime supports",
                dram.chip
            ))),
        }
    }
}

/// `placement`'s chips locked and joined, and its clusters reserved, before any boot.
pub(crate) struct Reserved {
    pub(crate) group: Group,
    /// Every chip's clusters, chip after chip, each running its bootloader.
    pub(crate) clusters: Vec<Bootloader>,
}

const FIRMWARE: &[u8] = include_bytes!(env!("FURIOSA_OPT_DEVICE_IMAGE"));
const _: () = assert!(FIRMWARE.len() > 8, "the firmware image opens with its magic word");
const BOOTLOADER: &[u8] = include_bytes!(env!("FURIOSA_OPT_BOOTLOADER"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, ToPlacements};

    /// An explicit placement resolves to itself; no placement at all, or one that is not ascending
    /// chips with all PEs when several, opens no device.
    #[test]
    fn rejects_placements_that_are_not_devices() {
        let placement = Placement {
            chips: vec![3, 1],
            pes: 4..8,
        };
        assert_eq!(placement.to_placements().unwrap(), vec![placement]);
        // A topology the installed chips cannot hold has no placement.
        assert!(Placement::place((3, 8).into(), vec![0, 1]).is_empty());
        assert!(matches!(Device::new(&[] as &[Placement]), Err(Error::Invalid(_))));
        for (chips, pes) in [(vec![], 0..8), (vec![3, 2], 0..8), (vec![0, 1], 0..4)] {
            let open = Driver::open(Placement { chips, pes }, Waker::noop().clone());
            assert!(matches!(open, Err(Error::Invalid(_))));
        }
    }
}
