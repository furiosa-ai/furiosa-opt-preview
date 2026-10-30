use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::task::{Poll, Waker};

use crate::arch::rngd;
use crate::buffer::{Allocator, Buffer};
use crate::driver::Driver;
use crate::function::Launches;
use crate::memory::{Kind, Usage};
use crate::pinned::Registration;
use crate::{Error, Pinned, Placement, Result, ToPlacements, Topology};

mod transfer;
mod worker;
pub(crate) use transfer::Transfers;
pub(crate) use worker::Worker;

/// The chips a process opened: functions load and launch through it, and it carries every
/// allocation and host transfer.
pub struct Device {
    /// Joined first, so admitted work drains before the driver it drives closes.
    pub(crate) worker: Worker,
    /// Every function's launch slots, so launches nobody polls still complete and run their
    /// epilogues.
    pub(crate) launches: Arc<Launches>,
    /// The transfers of pinned memory, so those nobody polls still complete.
    pub(crate) transfers: Arc<Transfers>,
    pub(crate) driver: Arc<Driver>,
    /// Which device bytes each Buffer owns; outlives the device through its Buffers.
    pub(crate) allocator: Arc<Allocator>,
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
            let allocator = Arc::new(Allocator::over(driver.topology, driver.allocatable.clone()));
            let driver = Arc::new(driver);
            let launches = Arc::<Launches>::default();
            let transfers = Arc::new(Transfers::new(waker.clone()));
            let work = Work {
                launches: Arc::clone(&launches),
                transfers: Arc::clone(&transfers),
                driver: Arc::clone(&driver),
            };
            return Ok(Self {
                worker: Worker::spawn(work, unpark),
                launches,
                transfers,
                driver,
                allocator,
                placement,
            });
        }
        Err(reserved.unwrap_or_else(|| Error::Invalid("no placement candidates".into())))
    }

    /// Allocates `len` bytes of `kind` memory at one address on every chip.
    ///
    /// - `Dram` takes, among the free spans that fit, the lowest one that puts the fewest bytes in
    ///   the peer window at the bottom of DRAM; `DramShared` takes the lowest one inside it.
    /// - SRAM residents fill down from the top, while each admitted launch's temporaries take the
    ///   SRAM below its image's `sram_end`. A resident that would start below those temporaries
    ///   fails with [`Error::SramOverlap`], as a launch does whose temporaries would reach the lowest
    ///   resident.
    /// - Dropped bytes are free again once their last owner, a running launch or transfer included,
    ///   lets go of them; [`Error::OutOfMemory`] when no free span fits.
    pub fn alloc(&self, kind: Kind, len: usize) -> Result<Buffer> {
        self.allocator.alloc(kind, len)
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
        if pinned
            .registration()
            .is_some_and(|registration| registration.belongs_to(&self.allocator))
        {
            return Ok(());
        }
        let len = u32::try_from(pinned.len())
            .map_err(|_| Error::Invalid("a mapped host allocation exceeds 4 GiB - 1 byte".into()))?;
        let invalid = || Error::Device("the driver's DMA mapping does not cover the host allocation".into());
        let tables = self
            .driver
            .register(pinned.dmabuf().fd().as_raw_fd())?
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
        pinned.set_registration(Some(Arc::new(Registration { tables: table, offsets })));
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
        self.allocator.topology()
    }

    /// The physical chips and PEs this device opened.
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Bytes per chip that `kind` allocations can occupy, live or free: what the driver and firmware
    /// leave, rounded down to whole allocation units.
    pub fn capacity(&self, kind: Kind) -> usize {
        self.allocator.capacity(kind)
    }

    /// Free `kind` memory per chip now, as [`Device::alloc`] of that kind can take it.
    pub fn usage(&self, kind: Kind) -> Usage {
        self.allocator.usage(kind)
    }

    /// The launches and transfers this device admitted.
    pub(crate) fn work(&self) -> Work {
        Work {
            launches: Arc::clone(&self.launches),
            transfers: Arc::clone(&self.transfers),
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
        let mailbox = (mailboxes.get_mut(self.cluster)).ok_or_else(crate::driver::dma::stopped)?;
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
    transfers: Arc<Transfers>,
    driver: Arc<Driver>,
}

impl Work {
    /// Advances the admitted transfers, then [`Self::collect`]s; true when none of either is in
    /// flight.
    pub(crate) fn complete(&self, contention: Contention) -> bool {
        self.transfers.poll() & self.collect(contention)
    }

    /// Collects the launches and DMA the hardware finished, running their epilogues; true when
    /// none is in flight. A failed device finishes no launch any more, so its launches count as
    /// done. A transfer waits with this, never [`Self::complete`], so no transfer runs inside
    /// another.
    pub(crate) fn collect(&self, contention: Contention) -> bool {
        let launches = self.launches.complete(contention) || self.driver.poison.get().is_some();
        self.driver.dma.complete() && launches
    }

    /// Ends once every launch and transfer admitted before this call ended, epilogues included;
    /// fails with the device's first failure once it failed.
    fn drained(&self) -> impl Future<Output = Result<()>> + Send + use<> {
        let launched = self.launches.launched();
        // Jobs first: a slot a job submits after this is covered by the job's end.
        let jobs = self.transfers.submitted().into_iter().map(Pending::Job);
        let slots = (self.driver.dma.submitted()).map(|(index, sequence)| Pending::Slot { index, sequence });
        let submitted = jobs.chain(slots).collect::<Vec<_>>();
        let (transfers, driver) = (Arc::clone(&self.transfers), Arc::clone(&self.driver));
        std::future::poll_fn(move |context| {
            if let Some(error) = driver.poison.get() {
                return Poll::Ready(Err(error.clone()));
            }
            let ended = |pending: &Pending| match *pending {
                Pending::Job(sequence) => transfers.ended(sequence, context.waker()),
                Pending::Slot { index, sequence } => driver.dma.ended(index, sequence, context.waker()),
            };
            if launched.ended(context.waker()) && submitted.iter().all(ended) {
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

/// A transfer admitted before a [`Device::synchronize`], which waits for its end.
enum Pending {
    /// A job of [`Transfers`], by admission number.
    Job(u64),
    /// A submitted DMA queue slot, by index and generation.
    Slot { index: usize, sequence: u64 },
}

/// Where a cluster's host descriptor table may start: `rngd::dma::Operand` stores the table
/// address in 256-byte units.
const TABLE_ALIGNMENT: usize = 256;
