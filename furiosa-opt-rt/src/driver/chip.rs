//! One chip of a device: its reservation, hardware identity, peer windows and firmware PDMA.

use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};

use nix::fcntl::{Flock, FlockArg};

use super::map::open_rw;
use super::window::Window;
use crate::arch::rngd::{self, CAPS_DMABUF, CAPS_RESERVE, GroupConfig, Peer, dma::Config, node};
use crate::{Error, Result};

/// One chip of the device, in rank order.
pub(crate) struct Chip {
    pub(crate) index: u8,
    pub(crate) physical: u8,
    /// Where this chip sees each peer's memory, by rank.
    pub(crate) peers: [u64; 8],
    /// Where PDMA and the chip's clusters address the device's share of its memory, once a boot
    /// found it.
    pub(crate) base: u64,
    /// BAR4, for host writes too small to pay a DMA round trip.
    pub(crate) window: Window,
    /// Firmware PDMA channels, held from initialization and released before the lock; none when
    /// the driver cannot reserve them.
    pub(crate) dma: Option<Dma>,
    /// The chip's dmar node, locked for this process and joined to its peers.
    pub(crate) lock: Flock<File>,
}

impl Chip {
    /// Locks and joins `chips` as one device. The second value is how much DRAM every peer
    /// window covers.
    pub(crate) fn join(chips: &[u8]) -> Result<(Vec<Self>, usize)> {
        let mode = if chips.len() > 1 {
            FlockArg::LockExclusiveNonblock
        } else {
            FlockArg::LockSharedNonblock
        };
        let locks = chips
            .iter()
            .map(|&chip| {
                Flock::lock(open_rw(&node(chip, "dmar"))?, mode).map_err(|(_, error)| Error::reserved(chip, error))
            })
            .collect::<Result<Vec<_>>>()?;
        let physical = chips
            .iter()
            .map(|&chip| {
                let mgmt = open_rw(&node(chip, "mgmt"))?;
                // SAFETY: the ioctl returns the hardware identity and takes an empty payload.
                let id = unsafe { rngd::physical_id(mgmt.as_raw_fd(), &mut ()) }
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
            .map(|(&chip, dmar)| Self::connect(chip, chips, &physical, dmar))
            .collect::<Result<Vec<_>>>()?;
        let shared_end = connected.iter().map(|(limit, _)| *limit).min().unwrap_or(0);
        let chips = chips
            .iter()
            .zip(physical)
            .zip(locks.into_iter().zip(connected))
            .map(|((&index, physical), (lock, (_, peers)))| {
                Ok(Self {
                    index,
                    physical,
                    peers,
                    base: 0,
                    window: Window::new(open_rw(&node(index, "bar4"))?),
                    dma: None,
                    lock,
                })
            })
            .collect::<Result<_>>()?;
        Ok((chips, shared_end))
    }

    /// Takes the firmware PDMA channels of the chip ranked `rank` among `count`, when the driver
    /// can reserve them; without them every transfer runs on the driver's own channels.
    pub(crate) fn open_dma(&mut self, rank: usize, count: usize) -> Result<Option<&Dma>> {
        let chip = self.index;
        let channel = open_rw(&node(chip, "ch0"))?;
        // SAFETY: this channel ioctl takes no userspace payload.
        let caps = unsafe { rngd::get_caps(channel.as_raw_fd()) }
            .map_err(|why| Error::Device(format!("asking npu{chip} PDMA capabilities: {why}")))?;
        match Dma::reserves(caps) {
            Capability::Unsupported => {
                return Err(Error::Device(format!(
                    "npu{chip} PDMA requires driver dma-buf mapping support"
                )));
            }
            Capability::Host => return Ok(None),
            Capability::Firmware => {}
        }
        let entry = count + (rngd::ATU_ENTRIES - count) / count * rank;
        Ok(Some(self.dma.insert(Dma::open(chip, &self.lock, entry as u16)?)))
    }

    fn connect(chip: u8, chips: &[u8], physical: &[u8], dmar: &Flock<File>) -> Result<(usize, [u64; 8])> {
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
        unsafe { rngd::set_group(dmar.as_raw_fd(), &mut group) }
            .map_err(|why| Error::Device(format!("npu{chip} joining {chips:?}: {why}")))?;
        if chips.len() == 1 {
            return Ok((0, [0; 8]));
        }
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
        Ok((limit, addresses))
    }
}

pub(crate) struct Dma {
    pub(crate) channels: Vec<File>,
    pub(crate) dmar: File,
    pub(crate) entry: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Capability {
    Unsupported,
    Host,
    Firmware,
}

impl Dma {
    fn reserves(caps: i32) -> Capability {
        match (caps & CAPS_DMABUF != 0, caps & CAPS_RESERVE != 0) {
            (false, _) => Capability::Unsupported,
            (true, false) => Capability::Host,
            (true, true) => Capability::Firmware,
        }
    }

    pub(crate) fn open(chip: u8, dmar: &File, entry: u16) -> Result<Self> {
        let channels = (4..8)
            .map(|channel| {
                let file = open_rw(&node(chip, &format!("ch{channel}")))?;
                // SAFETY: the fd identifies the channel reserved until its last close.
                unsafe { rngd::reserve(file.as_raw_fd()) }
                    .map_err(|error| Error::Device(format!("reserving DMA channel {channel}: {error}")))?;
                Ok(file)
            })
            .collect::<Result<_>>()?;
        let dmar = dmar
            .try_clone()
            .map_err(|error| Error::Device(format!("retaining DMA mapping: {error}")))?;
        let bar = open_rw(&node(chip, "bar0"))?;
        let mut info = rngd::BarInfo::default();
        // SAFETY: the driver writes two u64 fields describing this BAR.
        unsafe { rngd::bar_info(bar.as_raw_fd(), &mut info) }
            .map_err(|error| Error::Device(format!("querying DMA registers: {error}")))?;
        if info.size <= rngd::PDMA_EXTENT || info.size > 1 << 32 {
            return Err(Error::Device("DMA registers do not fit one oATU entry".into()));
        }
        let mut region = rngd::Region {
            offset: 0,
            size: info.size,
            fd: -1,
        };
        // SAFETY: the driver exports the validated BAR range into a new owned fd.
        unsafe { rngd::export(bar.as_raw_fd(), &mut region) }
            .map_err(|error| Error::Device(format!("exporting DMA registers: {error}")))?;
        // SAFETY: successful export transfers ownership of the returned fd.
        let buffer = unsafe { File::from_raw_fd(region.fd) };
        let mut mapping = rngd::Entry {
            fd: buffer.as_raw_fd(),
            offset: 0,
            index: entry,
            ats: 0,
        };
        // SAFETY: the group chose an unused entry; the driver retains the exported buffer.
        unsafe { rngd::set_entry(dmar.as_raw_fd(), &mut mapping) }
            .map_err(|error| Error::Device(format!("mapping DMA registers: {error}")))?;
        Ok(Self { channels, dmar, entry })
    }

    pub(crate) fn config(&self, scratch: u64, peers: [u64; 8], members: u8, column: usize, count: usize) -> Config {
        Config {
            registers: rngd::PDMA_REGISTERS + (u64::from(self.entry) << 32),
            // The host retains rngd::PDMA_DESCRIPTORS bytes for the firmware's descriptor chains.
            descriptors: scratch,
            channels: rngd::channels(column, count),
            local: rngd::LOCAL_DRAM,
            peers,
            members,
        }
    }

    pub(crate) fn channels(&self, mask: u8) -> [RawFd; 4] {
        std::array::from_fn(|index| {
            if mask & (1 << (index + 4)) == 0 {
                -1
            } else {
                self.channels[index].as_raw_fd()
            }
        })
    }
}

impl Drop for Dma {
    fn drop(&mut self) {
        // Closing reserved channels stops their engines before the mapping is released.
        self.channels.clear();
        // SAFETY: this object retains the dmar fd and owns the configured entry.
        let _ = unsafe { rngd::unset_entry(self.dmar.as_raw_fd(), &mut self.entry) };
    }
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
