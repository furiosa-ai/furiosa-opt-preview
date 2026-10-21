//! One chip of a device: its reservation, hardware identity, peer windows and firmware PDMA.

use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};

use nix::fcntl::{Flock, FlockArg};

use super::kernel::window::Window;
use super::kernel::{self, CAPS_DMABUF, CAPS_RESERVE, GroupConfig, Peer, node};
use crate::arch::rngd::{self, dma::Config};
use crate::{Error, Result};

/// One chip of the device, in rank order.
pub(crate) struct Chip {
    pub(crate) index: u8,
    /// Where this chip sees each peer's memory, by rank.
    pub(crate) peers: [u64; 8],
    /// Where PDMA and the chip's clusters address the device's share of its memory.
    pub(crate) base: u64,
    /// BAR4, for host writes too small to pay a DMA round trip.
    pub(crate) window: Window,
    /// The chip's dmar node, locked for this process and joined to its peers.
    pub(crate) lock: Flock<File>,
}

/// The chips of one device, locked and joined, before a boot found where their clusters see
/// memory.
pub(crate) struct Group {
    /// In rank order.
    pub(crate) chips: Vec<Joined>,
    /// Where the DRAM each chip shares with all its peers ends: the shortest of their windows.
    pub(crate) shared_end: usize,
}

impl Group {
    /// Locks and joins `chips` as one device.
    pub(crate) fn join(chips: &[u8]) -> Result<Self> {
        let mode = if chips.len() > 1 {
            FlockArg::LockExclusiveNonblock
        } else {
            FlockArg::LockSharedNonblock
        };
        let locks = chips
            .iter()
            .map(|&chip| {
                Flock::lock(
                    File::options()
                        .read(true)
                        .write(true)
                        .open(node(chip, "dmar"))
                        .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "dmar"))))?,
                    mode,
                )
                .map_err(|(_, error)| Error::reserved(chip, error))
            })
            .collect::<Result<Vec<_>>>()?;
        let physical = chips
            .iter()
            .map(|&chip| {
                let mgmt = File::options()
                    .read(true)
                    .write(true)
                    .open(node(chip, "mgmt"))
                    .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "mgmt"))))?;
                // SAFETY: the ioctl returns the hardware identity and takes an empty payload.
                let id = unsafe { kernel::physical_id(mgmt.as_raw_fd(), &mut ()) }
                    .map_err(|error| Error::Device(format!("npu{chip} physical identity: {error}")))?;
                u8::try_from(id).map_err(|_| Error::Invalid(format!("npu{chip} has invalid physical identity {id}")))
            })
            .collect::<Result<Vec<_>>>()?;
        if physical
            .iter()
            .enumerate()
            .any(|(rank, id)| physical[..rank].contains(id))
        {
            return Err(Error::Invalid("chips have duplicate physical identities".into()));
        }
        let connected = chips
            .iter()
            .zip(&locks)
            .map(|(&chip, dmar)| PeerWindows::join(chip, chips, &physical, dmar))
            .collect::<Result<Vec<_>>>()?;
        let shared_end = connected.iter().map(|windows| windows.end).min().unwrap_or(0);
        let chips = chips
            .iter()
            .zip(physical)
            .zip(locks.into_iter().zip(connected))
            .map(|((&index, physical), (lock, PeerWindows { addresses: peers, .. }))| {
                Ok(Joined {
                    index,
                    physical,
                    peers,
                    window: Window::new(
                        File::options()
                            .read(true)
                            .write(true)
                            .open(node(index, "bar4"))
                            .map_err(|why| Error::Device(format!("{}: {why}", node(index, "bar4"))))?,
                    ),
                    lock,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { chips, shared_end })
    }
}

/// One chip of a [`Group`].
pub(crate) struct Joined {
    pub(crate) index: u8,
    pub(crate) physical: u8,
    /// Where this chip sees each peer's memory, by rank.
    pub(crate) peers: [u64; 8],
    /// BAR4, which stages the firmware image.
    pub(crate) window: Window,
    /// The chip's dmar node, locked for this process and joined to its peers.
    pub(crate) lock: Flock<File>,
}

impl Joined {
    /// The chip, its clusters seeing its memory at `base`.
    pub(crate) fn open(self, base: u64) -> Chip {
        let Self {
            index,
            peers,
            window,
            lock,
            ..
        } = self;
        Chip {
            index,
            peers,
            base,
            window,
            lock,
        }
    }
}

/// The firmware's PCIe DMA: PDMA channels the kernel driver reserved for it on every chip of the
/// device. A device has it on all its chips or on none.
pub(crate) struct Pdma {
    /// In rank order.
    pub(crate) chips: Vec<Dma>,
}

impl Pdma {
    /// Reserves PDMA channels on every one of `chips`; `None` when the kernel driver cannot, and
    /// every transfer then runs on the driver's own channels.
    pub(crate) fn reserve(chips: &[Chip]) -> Result<Option<Self>> {
        let reserved = (chips.iter().enumerate())
            .map(|(rank, chip)| Dma::reserve(chip.index, &chip.lock, rank, chips.len()))
            .collect::<Result<Option<Vec<_>>>>()?;
        Ok(reserved.map(|chips| Self { chips }))
    }

    /// What the firmware of `chip`'s column `column` of `clusters` drives, with `scratch` for its
    /// descriptor chains.
    pub(crate) fn config(&self, chip: &Chip, rank: usize, scratch: u64, column: usize, clusters: usize) -> Config {
        let members = self.chips.len() as u8;
        self.chips[rank].config(scratch, chip.peers, members, column, clusters)
    }

    /// The channels the firmware of the chip ranked `rank`, column `column` of `clusters`, runs on.
    pub(crate) fn channels(&self, rank: usize, column: usize, clusters: usize) -> [RawFd; 4] {
        self.chips[rank].cluster(column, clusters).fds
    }
}

/// The PDMA channels a chip reserves for its firmware, channels 4 to 7, which its clusters share.
pub(crate) const FIRMWARE_CHANNELS: usize = 4;

pub(crate) struct Dma {
    pub(crate) channels: Vec<File>,
    pub(crate) dmar: File,
    pub(crate) entry: u16,
}

impl Dma {
    /// The firmware PDMA channels of the chip ranked `rank` among `count`, when the kernel driver can
    /// reserve them.
    pub(crate) fn reserve(chip: u8, lock: &Flock<File>, rank: usize, count: usize) -> Result<Option<Self>> {
        let channel = File::options()
            .read(true)
            .write(true)
            .open(node(chip, "ch0"))
            .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "ch0"))))?;
        // SAFETY: this channel ioctl takes no userspace payload.
        let caps = unsafe { kernel::get_caps(channel.as_raw_fd()) }
            .map_err(|why| Error::Device(format!("asking npu{chip} PDMA capabilities: {why}")))?;
        match Self::reserves(caps) {
            Capability::Unsupported => {
                return Err(Error::Unsupported(format!(
                    "npu{chip}: the kernel driver cannot map dma-buf memory for PDMA (no NPU_PDMA_CAPS_DMABUF)"
                )));
            }
            Capability::Host => return Ok(None),
            Capability::Firmware => {}
        }
        let entry = count + (rngd::ATU_ENTRIES - count) / count * rank;
        Self::open(chip, lock, entry as u16).map(Some)
    }

    pub(crate) fn open(chip: u8, dmar: &File, entry: u16) -> Result<Self> {
        let channels = (4..8)
            .map(|channel| {
                let file = File::options()
                    .read(true)
                    .write(true)
                    .open(node(chip, &format!("ch{channel}")))
                    .map_err(|why| Error::Device(format!("{}: {why}", node(chip, &format!("ch{channel}")))))?;
                // SAFETY: the fd identifies the channel reserved until its last close.
                unsafe { kernel::reserve(file.as_raw_fd()) }
                    .map_err(|error| Error::Device(format!("reserving DMA channel {channel}: {error}")))?;
                Ok(file)
            })
            .collect::<Result<_>>()?;
        let dmar = dmar
            .try_clone()
            .map_err(|error| Error::Device(format!("retaining DMA mapping: {error}")))?;
        let bar = File::options()
            .read(true)
            .write(true)
            .open(node(chip, "bar0"))
            .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "bar0"))))?;
        let mut info = kernel::BarInfo::default();
        // SAFETY: the driver writes two u64 fields describing this BAR.
        unsafe { kernel::bar_info(bar.as_raw_fd(), &mut info) }
            .map_err(|error| Error::Device(format!("querying DMA registers: {error}")))?;
        if info.size <= rngd::PDMA_EXTENT || info.size > 1 << 32 {
            return Err(Error::Device("DMA registers do not fit one oATU entry".into()));
        }
        let mut region = kernel::Region {
            offset: 0,
            size: info.size,
            fd: -1,
        };
        // SAFETY: the driver exports the validated BAR range into a new owned fd.
        unsafe { kernel::export(bar.as_raw_fd(), &mut region) }
            .map_err(|error| Error::Device(format!("exporting DMA registers: {error}")))?;
        // SAFETY: successful export transfers ownership of the returned fd.
        let buffer = unsafe { File::from_raw_fd(region.fd) };
        let mut mapping = kernel::Entry {
            fd: buffer.as_raw_fd(),
            offset: 0,
            index: entry,
            ats: 0,
        };
        // SAFETY: the group chose an unused entry; the driver retains the exported buffer.
        unsafe { kernel::set_entry(dmar.as_raw_fd(), &mut mapping) }
            .map_err(|error| Error::Device(format!("mapping DMA registers: {error}")))?;
        Ok(Self { channels, dmar, entry })
    }

    /// What the firmware of the cluster at column `column` of `count` drives, with `scratch` for its
    /// descriptor chains.
    pub(crate) fn config(&self, scratch: u64, peers: [u64; 8], members: u8, column: usize, count: usize) -> Config {
        Config {
            registers: rngd::PDMA_REGISTERS + (u64::from(self.entry) << 32),
            // The host retains rngd::PDMA_DESCRIPTORS bytes for the firmware's descriptor chains.
            descriptors: scratch,
            channels: self.cluster(column, count).mask,
            local: rngd::LOCAL_DRAM,
            peers,
            members,
        }
    }

    /// The channels the cluster at column `column` of `count` on this chip runs on: all four with
    /// one cluster, a pair each with two.
    pub(crate) fn cluster(&self, column: usize, count: usize) -> ClusterChannels {
        let mask = if count == 1 { 0xf0 } else { 0x30 << (column * 2) };
        ClusterChannels {
            mask,
            fds: std::array::from_fn(|index| {
                if mask & (1 << (index + 4)) == 0 {
                    -1
                } else {
                    self.channels[index].as_raw_fd()
                }
            }),
        }
    }

    fn reserves(caps: i32) -> Capability {
        match (caps & CAPS_DMABUF != 0, caps & CAPS_RESERVE != 0) {
            (false, _) => Capability::Unsupported,
            (true, false) => Capability::Host,
            (true, true) => Capability::Firmware,
        }
    }
}

impl Drop for Dma {
    fn drop(&mut self) {
        // Closing reserved channels stops their engines before the mapping is released.
        self.channels.clear();
        // SAFETY: this object retains the dmar fd and owns the configured entry.
        let _ = unsafe { kernel::unset_entry(self.dmar.as_raw_fd(), &mut self.entry) };
    }
}

/// The firmware PDMA channels one cluster runs on, of the four its chip reserved.
pub(crate) struct ClusterChannels {
    /// Which of channels 4 to 7, one bit each from bit 4.
    pub(crate) mask: u8,
    /// Their descriptors, `-1` for each channel the cluster does not run on.
    pub(crate) fds: [RawFd; 4],
}

/// One chip's windows onto its peers' DRAM, as the kernel driver set them up when it joined the
/// chip to them: a chip reaches a peer's memory through a window of its PCIe translation entries.
struct PeerWindows {
    /// Where the chip sees each peer's DRAM, by rank.
    addresses: [u64; 8],
    /// Where the windows end in each peer's DRAM, which they map from its start.
    end: usize,
}

impl PeerWindows {
    /// Joins `chip` to its peers among `chips`, whose physical identities are `physical`, through its
    /// locked `dmar` node.
    fn join(chip: u8, chips: &[u8], physical: &[u8], dmar: &Flock<File>) -> Result<Self> {
        let mut group = GroupConfig {
            local: u32::from(chip),
            ats: 0,
            remotes: (chips.len() - 1) as u16,
            peers: [Peer::default(); _],
        };
        for (slot, (&peer, &physical)) in group
            .peers
            .iter_mut()
            .zip(chips.iter().zip(physical).filter(|(other, _)| **other != chip))
        {
            *slot = Peer {
                file: u32::from(peer),
                npu: u32::from(physical),
                ..Peer::default()
            };
        }

        // SAFETY: the descriptor is this call's own, and `group` is the writable ioctl payload.
        unsafe { kernel::set_group(dmar.as_raw_fd(), &mut group) }
            .map_err(|why| Error::Device(format!("npu{chip} joining {chips:?}: {why}")))?;
        if chips.len() == 1 {
            return Ok(Self {
                addresses: [0; 8],
                end: 0,
            });
        }
        // The chip's translation entries map its own DRAM first, then share the rest evenly among
        // its peers; a window reaches no further than its share, whatever the driver reports.
        let mut limit = (rngd::ATU_ENTRIES - chips.len()) / chips.len() * rngd::ATU_ENTRY;
        let mut addresses = [0; 8];
        let peers = group.peers;
        for peer in &peers[..usize::from(group.remotes)] {
            let rank = chips.iter().position(|&other| u32::from(other) == peer.file);
            let Some(rank) = rank.filter(|_| peer.dram != 0 && peer.size != 0) else {
                return Err(Error::Device(
                    "the driver returned an unknown or empty peer window".into(),
                ));
            };
            addresses[rank] = peer.dram;
            limit = limit.min(peer.size as usize);
        }
        Ok(Self { addresses, end: limit })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Capability {
    Unsupported,
    Host,
    Firmware,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The driver's PDMA capability bits: a device opens on dma-buf support, with firmware DMA
    /// when the driver can also reserve channels.
    #[test]
    fn reads_driver_capabilities() {
        let caps = [0, CAPS_RESERVE, CAPS_DMABUF, CAPS_DMABUF | CAPS_RESERVE];
        assert_eq!(
            caps.map(Dma::reserves),
            [
                Capability::Unsupported,
                Capability::Unsupported,
                Capability::Host,
                Capability::Firmware
            ]
        );
    }
}
