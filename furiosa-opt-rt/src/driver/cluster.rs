//! Cluster reservation, boot staging and raw queue mappings.

use super::map::{Mapping, open_rw};
use crate::arch::rngd::{self, BAR2_LEN, CQ_LEN, CQ_REGION, DOORBELLS, PeRange, node};
use crate::{Error, Result};
use furiosa_opt_abi::doorbell::Doorbells;
use furiosa_opt_abi::ring::{Consumer, Producer, Ring};
use nix::errno::Errno;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};

pub(crate) struct Cluster {
    pub(crate) submission: Producer,
    pub(crate) completion: Consumer,
    _bar2: Mapping,
    _page: Mapping,
    reservation: File,
    pub(crate) channels: [RawFd; 4],
}

impl Cluster {
    pub(crate) fn open(chip: u8, pes: PeRange, rank: u8, boot: &[u8]) -> Result<Self> {
        // Staging refuses PEs in use, so it goes first; the memory window must be set before the
        // node opens, which starts the PEs on the staged bootloader.
        Self::stage(chip, pes, boot)?;
        Self::map_dram(chip, pes)?;
        let cluster_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(node(chip, &pes.node()))
            .map_err(|why| Error::reserved(chip, Errno::from_raw(why.raw_os_error().unwrap_or(0))))?;
        let bars = open_rw(&node(chip, "bar2"))?;
        let bar2 = Mapping::new(&bars, (pes.window() + DOORBELLS.entries as u64) as i64, BAR2_LEN)?;
        let page = Mapping::new(&cluster_file, CQ_REGION, CQ_LEN).map_err(|why| {
            Error::Device(format!(
                "{why}; the driver must offer the completion-queue mapping (NPU_PE_MMAP_REGION_CQ)"
            ))
        })?;
        Self::new(bar2, page, cluster_file, pes.queue(rank))
    }

    /// Registers a dma-buf until this PE reservation closes. The driver stops firmware and
    /// its reserved channels before unmapping; the host does not submit individual transfers.
    pub(crate) fn map(&self, fd: RawFd) -> Result<Vec<rngd::dma::Segment>> {
        if self.channels.iter().all(|&fd| fd < 0) {
            return Err(Error::Device("firmware DMA is unavailable".into()));
        }
        let mut request = rngd::DmaMap {
            fd,
            capacity: 0,
            segments: 0,
            count: 0,
            reserved: 0,
            channels: self.channels,
            size: 0,
        };
        let mut segments = Vec::new();
        loop {
            request.capacity = segments.len() as u32;
            request.segments = segments.as_mut_ptr() as u64;
            // SAFETY: both ioctl payload and its segment array remain writable for this call.
            match unsafe { rngd::dma_map(self.reservation.as_raw_fd(), &mut request) } {
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

    fn map_dram(chip: u8, pes: PeRange) -> Result<()> {
        let mgmt = open_rw(&node(chip, "mgmt"))?;
        let mut window = rngd::Window {
            kind: rngd::WINDOW_DRAM,
            pes: pes.mask(),
            base: rngd::DEVICE_MEMORY.start + rngd::DRAM_FIRMWARE,
            size: rngd::DRAM,
        };
        // SAFETY: the request reads `window` and writes back only its own result.
        unsafe { rngd::set_window(mgmt.as_raw_fd(), &mut window) }
            .map(|_| ())
            .map_err(|why| Error::Device(format!("npu{chip} {} addressing memory: {why}", pes.node())))
    }

    /// Hands the signed bootloader to the driver, which verifies it and starts the cluster's PEs
    /// on it once their node opens.
    fn stage(chip: u8, pes: PeRange, image: &[u8]) -> Result<()> {
        let size = u32::try_from(image.len()).map_err(|_| Error::Device("bootloader exceeds 4 GiB".into()))?;
        let channel = open_rw(&node(chip, "ch0"))?;
        let mut stage = rngd::Stage {
            pe_ids: u32::from(pes.mask()),
            size,
            data: image.as_ptr() as u64,
        };
        // SAFETY: `stage` names this process's own buffer for the length it states.
        unsafe { rngd::stage(channel.as_raw_fd(), &mut stage) }
            .map(|_| ())
            .map_err(|why| Error::reserved(chip, why))
    }

    /// Takes over doorbell `queue`'s rings in `bar2` and the completion `page` of the node `file`.
    pub(crate) fn new(bar2: Mapping, page: Mapping, file: File, queue: u8) -> Result<Self> {
        // SAFETY: `bar2` maps the PE's doorbells from the first ring on, which the boot initialized.
        let doorbells = unsafe { Doorbells::at(bar2.at.as_ptr().wrapping_sub(DOORBELLS.entries), &DOORBELLS) };
        // SAFETY: `page` maps the cluster's completion ring.
        let completion = unsafe { Ring::completion(page.at.as_ptr(), rngd::COMPLETION_RING) };
        let why = |why| Error::Device(format!("the cluster's rings are not in a usable state: {why}"));
        Ok(Self {
            submission: doorbells.ring(queue).and_then(Producer::new).map_err(why)?,
            completion: Consumer::new(completion).map_err(why)?,
            _bar2: bar2,
            _page: page,
            reservation: file,
            channels: [-1; 4],
        })
    }
}
