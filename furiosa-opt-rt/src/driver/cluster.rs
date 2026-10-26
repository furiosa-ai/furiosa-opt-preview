//! Cluster reservation, boot staging and the mappings its link runs over.

use std::ops::Range;

use super::kernel::map::Mapping;
use super::kernel::{self, BAR2_LEN, CQ_LEN, CQ_REGION, PE_WINDOW, node};
use super::link::Link;
use crate::arch::rngd::{self, DOORBELLS, PES};
use crate::{Error, Result, Topology};
use nix::errno::Errno;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};

/// One cluster's PEs, reserved through their node until it closes, which stops them.
pub(crate) struct Cluster {
    reservation: File,
}

impl Cluster {
    /// Reserves `pes` on `chip` and starts them on the bootloader `boot`; returns the reservation
    /// and the link to the cluster's lead PE, which rank `rank`'s doorbell queue carries.
    pub(crate) fn open(chip: u8, pes: PeRange, rank: u8, boot: &[u8]) -> Result<(Self, Link)> {
        // Hand the signed bootloader to the driver, which verifies it and starts the cluster's PEs
        // on it once their node opens. Staging refuses PEs in use, so it goes first.
        let size = u32::try_from(boot.len()).map_err(|_| Error::Device("bootloader exceeds 4 GiB".into()))?;
        let channel = File::options()
            .read(true)
            .write(true)
            .open(node(chip, "ch0"))
            .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "ch0"))))?;
        let mut stage = kernel::Stage {
            pe_ids: u32::from(pes.mask()),
            size,
            data: boot.as_ptr() as u64,
        };
        // SAFETY: `stage` names this process's own buffer for the length it states.
        unsafe { kernel::stage(channel.as_raw_fd(), &mut stage) }.map_err(|why| Error::reserved(chip, why))?;
        // Point the PEs' memory window at the chip's DRAM: the PEs see the DRAM the firmware
        // and the device's memory use, past the driver's own reservation, at the device-memory
        // addresses device code is built for. It must be set before the node opens.
        let mgmt = File::options()
            .read(true)
            .write(true)
            .open(node(chip, "mgmt"))
            .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "mgmt"))))?;
        let mut window = kernel::PeWindow {
            kind: kernel::WINDOW_DRAM,
            pes: pes.mask(),
            base: rngd::DEVICE_MEMORY.start + rngd::DRAM_FIRMWARE,
            size: rngd::DRAM,
        };
        // SAFETY: the request reads `window` and writes back only its own result.
        unsafe { kernel::set_window(mgmt.as_raw_fd(), &mut window) }
            .map_err(|why| Error::Device(format!("npu{chip} {} addressing memory: {why}", pes.node())))?;
        let cluster_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(node(chip, &pes.node()))
            .map_err(|why| Error::reserved(chip, Errno::from_raw(why.raw_os_error().unwrap_or(0))))?;
        let bars = File::options()
            .read(true)
            .write(true)
            .open(node(chip, "bar2"))
            .map_err(|why| Error::Device(format!("{}: {why}", node(chip, "bar2"))))?;
        let bar2 = Mapping::new(&bars, (pes.window() + DOORBELLS.entries as u64) as i64, BAR2_LEN)?;
        let page = Mapping::new(&cluster_file, CQ_REGION, CQ_LEN).map_err(|why| {
            Error::Device(format!(
                "{why}; the driver must offer the completion-queue mapping (NPU_PE_MMAP_REGION_CQ)"
            ))
        })?;
        let link = Link::new(bar2, page, pes.queue(rank))?;
        Ok((Self::new(cluster_file), link))
    }

    /// The cluster whose PEs the node `reservation` holds.
    pub(crate) fn new(reservation: File) -> Self {
        Self { reservation }
    }

    /// Registers the dma-buf `fd` for the cluster's firmware PDMA `channels` until this PE
    /// reservation closes: the kernel driver pins its pages and maps them for the PDMA engine,
    /// and returns the bus addresses they got, in order. The pages are contiguous to the host but
    /// not on the bus, so the firmware turns an offset into the buffer into a bus address through
    /// these segments. The driver stops the firmware and its channels before it unmaps them.
    pub(crate) fn register(&self, fd: RawFd, channels: [RawFd; 4]) -> Result<Vec<rngd::dma::Segment>> {
        let mut request = kernel::DmaMap {
            fd,
            capacity: 0,
            segments: 0,
            count: 0,
            reserved: 0,
            channels,
            size: 0,
        };
        let mut segments = Vec::new();
        loop {
            request.capacity = segments.len() as u32;
            request.segments = segments.as_mut_ptr() as u64;
            // SAFETY: both ioctl payload and its segment array remain writable for this call.
            match unsafe { kernel::dma_map(self.reservation.as_raw_fd(), &mut request) } {
                Ok(_) => {
                    if request.count as usize > segments.len() || request.count == 0 {
                        return Err(Error::Device("driver returned an invalid DMA mapping".into()));
                    }
                    segments.truncate(request.count as usize);
                    return Ok(segments);
                }
                Err(Errno::ENOSPC) if request.count > request.capacity => {
                    segments.resize(request.count as usize, rngd::dma::Segment { address: 0, len: 0 });
                }
                Err(error) => return Err(Error::Device(format!("registering host DMA memory: {error}"))),
            }
        }
    }
}

/// The PEs one cluster fuses on a chip: `count` (1, 2 or 4) from `lead`, a multiple of `count`,
/// exactly the driver's fusion nodes. The lead PE's queue and control window are the cluster's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct PeRange {
    lead: u8,
    count: u8,
}

impl PeRange {
    /// The clusters that run `pes` on a chip: an aligned slice of 1, 2 or 4 PEs, or for all
    /// eight, both four-PE clusters.
    pub(crate) fn clusters(pes: Range<u8>) -> Result<Vec<Self>> {
        let count = pes.len() as u8;
        if !matches!(count, 1 | 2 | 4 | 8) || !pes.start.is_multiple_of(count) || pes.end > PES {
            return Err(Error::Invalid(format!(
                "invalid PEs {pes:?}: expected an aligned slice of 1, 2, 4 or 8"
            )));
        }
        let count = count / Topology { chips: 1, pes: count }.clusters() as u8;
        Ok(pes.step_by(count.into()).map(|lead| Self { lead, count }).collect())
    }

    /// The doorbell queue of this cluster on the chip ranked `member` in its device.
    pub(crate) fn queue(self, member: u8) -> u8 {
        member * PES + self.lead
    }

    pub(crate) fn window(self) -> u64 {
        u64::from(self.lead) * PE_WINDOW
    }

    /// The driver's bitmask of these PEs.
    pub(crate) fn mask(self) -> u8 {
        ((1 << self.count) - 1) << self.lead
    }

    pub(crate) fn node(self) -> String {
        match self.count {
            1 => format!("pe{}", self.lead),
            count => format!("pe{}-{}", self.lead, self.lead + count - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cluster is a kernel driver fusion node, with the driver's name, PE mask, queue and window.
    #[test]
    fn clusters_are_driver_fusion_nodes() {
        for (pes, nodes) in [
            (0..8, vec![(0x0f, 0, 0, "pe0-3"), (0xf0, 4, 0x8000_0000, "pe4-7")]),
            (4..8, vec![(0xf0, 4, 0x8000_0000, "pe4-7")]),
            (6..8, vec![(0xc0, 6, 0xc000_0000, "pe6-7")]),
            (5..6, vec![(0x20, 5, 0xa000_0000, "pe5")]),
        ] {
            let clusters = PeRange::clusters(pes).unwrap();
            assert_eq!(clusters.len(), nodes.len());
            for (cluster, node) in clusters.into_iter().zip(nodes) {
                let route = (cluster.mask(), cluster.queue(0), cluster.window(), &*cluster.node());
                assert_eq!(route, node);
            }
        }
        for pes in [0..0, 0..3, 2..6, 1..3, 8..16] {
            assert!(PeRange::clusters(pes.clone()).is_err(), "{pes:?}");
        }
    }
}
