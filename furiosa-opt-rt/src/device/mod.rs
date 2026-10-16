use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::task::{Poll, Waker};

use crate::arch::rngd;
use crate::buffer::{Allocations, Buffer};
use crate::driver::{Driver, NO_FIRMWARE_DMA};
use crate::memory::{Kind, Usage};
use crate::pinned::Mapping;
use crate::{Error, Pinned, Placement, Result, Sample, ToPlacements, Topology};

mod transfer;
mod worker;
pub(crate) use worker::Worker;

/// Where a cluster's host descriptor table may start: `rngd::dma::Operand` stores the table
/// address in 256-byte units.
const TABLE_ALIGNMENT: usize = 256;

/// The chips a process opened: functions load and launch through it, and it carries every
/// allocation and host transfer.
pub struct Device {
    /// Joined first, so admitted work drains before the driver it drives closes.
    pub(crate) worker: Worker,
    pub(crate) driver: Arc<Driver>,
    /// Which device bytes each Buffer owns; outlives the device through its Buffers.
    pub(crate) allocations: Arc<Allocations>,
    pub(crate) placement: Placement,
}

impl Device {
    /// Opens candidates in order, skipping only [`Error::Reserved`]; empty candidates are invalid.
    pub fn new(placements: impl ToPlacements) -> Result<Self> {
        let unpark = Arc::new(worker::Unpark::default());
        let waker = Waker::from(Arc::clone(&unpark));
        let mut reserved = None;
        for placement in placements.to_placements()? {
            let (driver, ranges) = match Driver::open(placement.clone(), waker.clone()) {
                Err(Error::Reserved(chip)) => {
                    reserved = Some(Error::Reserved(chip));
                    continue;
                }
                opened => opened?,
            };
            let allocations = Arc::new(Allocations::new(driver.topology, ranges));
            let driver = Arc::new(driver);
            return Ok(Self {
                worker: Worker::spawn(Arc::clone(&driver), unpark),
                driver,
                allocations,
                placement,
            });
        }
        Err(reserved.unwrap_or_else(|| Error::Invalid("no placement candidates".into())))
    }

    /// The physical chips and PEs this device opened.
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// One counter read of cluster `cluster`, in device rank order; `None` while the cluster has
    /// work in flight, which the read would wait behind.
    pub fn sample(&self, cluster: usize) -> Result<Option<Sample>> {
        self.driver.sample(cluster)
    }

    /// The chip count and processing elements per chip held by this device.
    pub fn topology(&self) -> Topology {
        self.allocations.topology
    }

    /// Bytes per chip that `kind` allocations can occupy, live or free: what the driver and firmware
    /// leave, rounded down to whole allocation units.
    pub fn capacity(&self, kind: Kind) -> usize {
        self.allocations.capacity(kind)
    }

    /// Free `kind` memory per chip now, as [`Device::alloc`] of that kind can take it.
    pub fn usage(&self, kind: Kind) -> Usage {
        self.allocations.usage(kind)
    }

    /// Runs `admit`; on [`Error::Busy`] retires the work this device finished and runs it once
    /// more. The caller must hold no transfer slot's lock.
    pub(crate) fn admit<T>(&self, mut admit: impl FnMut() -> Result<T>) -> Result<T> {
        match admit() {
            Err(Error::Busy) => {
                self.worker.launches.retire();
                self.driver.transfers.progress();
                admit()
            }
            result => result,
        }
    }

    /// Waits for every launch and transfer admitted before this call, epilogues included; the
    /// driver's first failure once the device failed.
    pub fn synchronize(&self) -> impl Future<Output = Result<()>> + Send + use<> {
        let launched = self.worker.launches.launched();
        let submitted = self.driver.transfers.submitted();
        let driver = Arc::clone(&self.driver);
        std::future::poll_fn(move |context| {
            if let Some(error) = driver.failure.get().or_else(|| driver.transfers.failure.get()) {
                return Poll::Ready(Err(error.clone()));
            }
            if launched.ended(context.waker()) && driver.transfers.ended(&submitted, context.waker()) {
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        })
    }

    /// Allocates `len` bytes of `kind` memory at one address on every chip.
    pub fn alloc(&self, kind: Kind, len: usize) -> Result<Buffer> {
        self.allocations.alloc(kind, len)
    }

    /// Registers pinned host memory for firmware-issued DMA on every cluster.
    ///
    /// Registration and descriptor upload happen once, before launch; launches then take its views
    /// as [`crate::Argument::Host`], and `write` and `read` move it on the clusters' channels. The
    /// driver retains the pinned pages until this device closes. Each mapped allocation is limited
    /// to `u32::MAX` bytes by the DMA operand encoding. [`Error::Device`] when the driver reserves
    /// no PDMA channels for the firmware; `write` and `read` then move the memory on the driver's.
    pub async fn map(&self, pinned: &mut Pinned) -> Result<()> {
        if !self.driver.firmware_dma() {
            return Err(Error::Device(NO_FIRMWARE_DMA.into()));
        }
        if let Some(mapping) = &pinned.mapping
            && mapping.table.belongs_to(&self.allocations)
        {
            return Ok(());
        }
        let len = u32::try_from(pinned.bytes)
            .map_err(|_| Error::Invalid("a mapped host allocation exceeds 4 GiB - 1 byte".into()))?;
        let invalid = || Error::Device("the driver's DMA mapping does not cover the host allocation".into());
        let tables = self
            .driver
            .register(pinned.allocation.fd.as_raw_fd())?
            .into_iter()
            .map(|segments| rngd::dma::Mapping::table(len.into(), segments).ok_or_else(invalid))
            .collect::<Result<Vec<_>>>()?;
        // A chip's clusters' tables lie one after another, at the same offsets on every chip.
        let clusters = tables.len() / self.driver.chips.len();
        let mut end = 0;
        let offsets = (tables.iter().enumerate())
            .map(|(cluster, words)| {
                let at = if cluster % clusters == 0 { 0 } else { end };
                end = (at + words.len() * size_of::<u64>()).next_multiple_of(TABLE_ALIGNMENT);
                at
            })
            .collect::<Vec<_>>();
        let size = offsets
            .iter()
            .zip(&tables)
            .map(|(at, words)| at + words.len() * size_of::<u64>())
            .max();
        let mut table = self.alloc(Kind::Dram, size.unwrap_or(0))?;
        for (cluster, (words, &at)) in tables.iter().zip(&offsets).enumerate() {
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            let view = table.slice_mut(at..at + bytes.len())?.on(cluster / clusters)?;
            self.write([(bytes.as_slice().into(), view)]).await?;
        }
        pinned.mapping = Some(Arc::new(Mapping { table, offsets }));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device reports the physical placement retained by its runtime owner.
    #[test]
    fn reports_opened_placement() {
        let device = crate::test::device(&[3, 1], 4);
        assert_eq!(
            device.placement(),
            &Placement {
                chips: vec![3, 1],
                pes: 0..4
            }
        );
    }

    /// Empty candidates fail before touching the driver.
    #[test]
    fn rejects_empty_placements() {
        assert!(matches!(Device::new(&[] as &[Placement]), Err(Error::Invalid(_))));
    }
}
