use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::task::{Poll, Waker};

use crate::arch::rngd;
use crate::buffer::{Allocations, Buffer};
use crate::driver::Driver;
use crate::function::Launches;
use crate::memory::{Kind, Usage};
use crate::pinned::Registration;
use crate::{Error, Pinned, Placement, Result, ToPlacements, Topology};

mod transfer;
mod worker;
pub(crate) use worker::Worker;

/// The chips a process opened: functions load and launch through it, and it carries every
/// allocation and host transfer.
pub struct Device {
    /// Joined first, so admitted work drains before the driver it drives closes.
    pub(crate) worker: Worker,
    /// Every function's launch slots, so launches nobody polls still complete and run their
    /// epilogues.
    pub(crate) launches: Arc<Launches>,
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
            let driver = match Driver::open(placement.clone(), waker.clone()) {
                Err(Error::Reserved(chip)) => {
                    reserved = Some(Error::Reserved(chip));
                    continue;
                }
                opened => opened?,
            };
            let allocations = Arc::new(Allocations::over(driver.topology, driver.allocatable.clone()));
            let driver = Arc::new(driver);
            let launches = Arc::<Launches>::default();
            let work = Work {
                launches: Arc::clone(&launches),
                driver: Arc::clone(&driver),
            };
            return Ok(Self {
                worker: Worker::spawn(work, unpark),
                launches,
                driver,
                allocations,
                placement,
            });
        }
        Err(reserved.unwrap_or_else(|| Error::Invalid("no placement candidates".into())))
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
    /// to `u32::MAX` bytes by the DMA operand encoding. [`Error::Unsupported`] when the driver
    /// reserves no PDMA channels for the firmware; `write` and `read` then move the memory on the driver's.
    pub async fn map(&self, pinned: &mut Pinned) -> Result<()> {
        self.driver.pdma()?;
        if let Some(registration) = &pinned.registration()
            && registration.buffer.belongs_to(&self.allocations)
        {
            return Ok(());
        }
        let len = u32::try_from(pinned.len())
            .map_err(|_| Error::Invalid("a mapped host allocation exceeds 4 GiB - 1 byte".into()))?;
        let invalid = || Error::Device("the driver's DMA mapping does not cover the host allocation".into());
        let tables = self
            .driver
            .register(pinned.allocation().fd().as_raw_fd())?
            .into_iter()
            .map(|segments| rngd::dma::SegmentTable::encode(len.into(), segments).ok_or_else(invalid))
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
        pinned.set_registration(Some(Arc::new(Registration {
            buffer: table,
            tables: offsets,
        })));
        Ok(())
    }

    /// Waits for every launch and transfer admitted before this call, epilogues included; the
    /// driver's first failure once the device failed.
    pub fn synchronize(&self) -> impl Future<Output = Result<()>> + Send + use<> {
        self.work().drained()
    }

    /// Cluster `cluster`'s clock, in device rank order.
    pub fn clock(&self, cluster: usize) -> Result<Clock<'_>> {
        if cluster >= self.topology().clusters() {
            return Err(Error::Invalid(format!("the device has no cluster {cluster}")));
        }
        Ok(Clock { device: self, cluster })
    }

    /// The chip count and processing elements per chip held by this device.
    pub fn topology(&self) -> Topology {
        self.allocations.topology()
    }

    /// The physical chips and PEs this device opened.
    pub fn placement(&self) -> &Placement {
        &self.placement
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

    /// The launches and transfers this device admitted.
    pub(crate) fn work(&self) -> Work {
        Work {
            launches: Arc::clone(&self.launches),
            driver: Arc::clone(&self.driver),
        }
    }

    /// Runs `admit`; on [`Error::Busy`] collects the work this device finished and runs it once
    /// more. The caller must hold no transfer slot's lock.
    pub(crate) fn admit<T>(&self, mut admit: impl FnMut() -> Result<T>) -> Result<T> {
        match admit() {
            Err(Error::Busy) => {
                self.work().complete(Contention::Wait);
                admit()
            }
            result => result,
        }
    }
}

/// One cluster's ARM counter, read against the host's clock.
pub struct Clock<'d> {
    device: &'d Device,
    cluster: usize,
}

impl Clock<'_> {
    /// The counter between two host clock reads; `None` while the cluster has work in flight,
    /// which the read would wait behind.
    pub fn read(&self) -> Result<Option<ClockReading>> {
        let host = || {
            let mut time = nix::libc::timespec { tv_sec: 0, tv_nsec: 0 };
            // SAFETY: `time` is writable storage for the requested monotonic clock.
            if unsafe { nix::libc::clock_gettime(nix::libc::CLOCK_MONOTONIC_RAW, &mut time) } != 0 {
                return Err(Error::Device(std::io::Error::last_os_error().to_string()));
            }
            Ok(time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64)
        };
        let mut mailboxes = self.device.driver.mailboxes.lock().unwrap();
        let mailbox = &mut mailboxes[self.cluster];
        if !mailbox.idle() {
            return Ok(None);
        }
        let before = host()?;
        let counter = (mailbox.call(&furiosa_opt_ipc::Clock)?)
            .map_err(|failure| crate::driver::mailbox::refused::<furiosa_opt_ipc::Clock>(self.cluster, &failure))?;
        let after = host()?;
        Ok(Some(ClockReading { before, counter, after }))
    }
}

/// A cluster's ARM counter in ticks, read between two host `CLOCK_MONOTONIC_RAW` readings in ns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockReading {
    pub before: u64,
    pub counter: u64,
    pub after: u64,
}

/// What collecting a device's work does with a launch slot another thread is advancing now.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Contention {
    /// Skip it: the worker never waits for a slot that an admission is binding.
    Skip,
    /// Wait for it, so every launch the hardware finished releases its memory before this returns.
    Wait,
}

/// The launches and transfers a device admitted.
pub(crate) struct Work {
    launches: Arc<Launches>,
    driver: Arc<Driver>,
}

impl Work {
    /// Collects the launches and transfers the hardware finished, running their epilogues; true
    /// when none is in flight. A failed device finishes no launch any more, so its launches count
    /// as done.
    pub(crate) fn complete(&self, contention: Contention) -> bool {
        let launches = self.launches.complete(contention) || self.driver.poison.get().is_some();
        self.driver.transfers.complete() && launches
    }

    /// Ends once every launch and transfer admitted before this call ended, epilogues included;
    /// fails with the device's first failure once it failed.
    fn drained(&self) -> impl Future<Output = Result<()>> + Send + use<> {
        let launched = self.launches.launched();
        let submitted = self.driver.transfers.submitted();
        let driver = Arc::clone(&self.driver);
        std::future::poll_fn(move |context| {
            if let Some(error) = driver.poison.get() {
                return Poll::Ready(Err(error.clone()));
            }
            if launched.ended(context.waker()) && driver.transfers.ended(&submitted, context.waker()) {
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        })
    }

    /// Stops task code on every cluster, before its DMA channels or peer mappings are released.
    fn stop(&self) {
        self.driver.stop();
    }
}

/// Where a cluster's host descriptor table may start: `rngd::dma::Operand` stores the table
/// address in 256-byte units.
const TABLE_ALIGNMENT: usize = 256;
