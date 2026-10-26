//! Host transfers through queue slots that callers submit and any poller publishes and completes.

use std::collections::BTreeSet;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Waker};

use arrayvec::ArrayVec;
use smallvec::SmallVec;

use crate::arch::rngd;
use crate::buffer::{Admission, Lease, Region, Waiters};
use crate::device::{Contention, Device};
use crate::driver::dma::{Command, Copy, Direction, Host, Route};
use crate::memory::Kind;
use crate::pinned::{Backing, DmaBuf, Registration};
use crate::{Access, BufferSlice, BufferSliceMut, Error, HostBufferSlice, HostBufferSliceMut, Result, Topology};

impl Device {
    /// Copies host bytes into exclusive device views once the launches and transfers admitted before
    /// it that use those bytes ended. Selected chips consume host bytes in rank order. Ordinary host
    /// slices finish synchronously after submission, including for forgotten futures.
    pub async fn write(
        &self,
        pairs: impl IntoIterator<Item = (HostBufferSlice<'_>, BufferSliceMut<'_>)>,
    ) -> Result<()> {
        let pairs = (pairs.into_iter())
            .map(|(host, view)| {
                let range = host.address..host.address + host.len;
                Pair {
                    view: view.0,
                    host: range,
                    pinned: host.pinned,
                    source: if host.pinned.is_none() {
                        // SAFETY: an ordinary write view borrows these initialized bytes.
                        Some(unsafe {
                            std::slice::from_raw_parts(std::ptr::with_exposed_provenance(host.address), host.len)
                        })
                    } else {
                        None
                    },
                }
            })
            .collect::<SmallVec<[_; 4]>>();
        self.transfer(Direction::Write, pairs).await
    }

    /// Copies shared device views into exclusive host views once the launches and transfers admitted
    /// before it that write those views ended; byte lengths must match exactly.
    pub async fn read(&self, pairs: impl IntoIterator<Item = (BufferSlice<'_>, HostBufferSliceMut<'_>)>) -> Result<()> {
        let pairs = (pairs.into_iter())
            .map(|(view, host)| {
                let range = host.address..host.address + host.len;
                Pair {
                    view,
                    host: range,
                    pinned: host.pinned,
                    source: None,
                }
            })
            .collect::<SmallVec<[_; 4]>>();
        self.transfer(Direction::Read, pairs).await
    }

    async fn transfer(&self, direction: Direction, pairs: SmallVec<[Pair<'_>; 4]>) -> Result<()> {
        let context = TransferContext {
            driver: Arc::clone(&self.driver),
            allocator: Arc::clone(&self.allocator),
            work: self.work(),
        };
        if direction == Direction::Write
            && pairs.iter().map(|pair| pair.host.len()).sum::<usize>() <= MMIO_WRITE_LIMIT
            && context.inline(&pairs)?
        {
            return Ok(());
        }
        let access = context.acquire(direction, &pairs)?;
        if pairs.iter().any(|pair| pair.pinned.is_none()) {
            // Borrowed bytes have no lease to keep them alive, so a forgotten future would leave the
            // device copying freed memory: the transfer ends inside this call, which progresses the
            // device itself rather than return the Pending its waits would yield.
            let mut transfer = std::pin::pin!(context.run(direction, &pairs, access));
            loop {
                if let std::task::Poll::Ready(result) = transfer.as_mut().poll(&mut Context::from_waker(Waker::noop()))
                {
                    return result;
                }
                self.work().complete(Contention::Wait);
                std::thread::yield_now();
            }
        }
        let retained = pairs.iter().map(Pair::retain).collect::<Vec<_>>();
        let result = Arc::new(std::sync::Mutex::new((None, None::<Waker>)));
        let completed = Arc::clone(&result);
        self.transfers.enqueue(async move {
            let pairs = retained.iter().map(Retained::pair).collect::<SmallVec<[_; 4]>>();
            let output = context.run(direction, &pairs, access).await;
            let mut completed = completed.lock().unwrap();
            completed.0 = Some(output);
            if let Some(waker) = completed.1.take() {
                waker.wake();
            }
        });
        self.work().complete(Contention::Wait);
        std::future::poll_fn(move |context| {
            let mut result = result.lock().unwrap();
            match result.0.take() {
                Some(result) => std::task::Poll::Ready(result),
                None => {
                    result.1 = Some(context.waker().clone());
                    std::task::Poll::Pending
                }
            }
        })
        .await
    }
}

/// The transfers of pinned memory a device admitted, each run to completion whether or not its
/// caller still observes it: it waits for earlier work and queue room, and submits its slots in
/// turn.
pub(crate) struct Transfers {
    /// Jobs admitted since the poller last took them.
    incoming: Mutex<Vec<(u64, Job)>>,
    /// The jobs being run. Only the thread holding this polls them, and it holds no other lock of
    /// this type meanwhile, so a job may admit another transfer while it runs.
    running: Mutex<Vec<(u64, Job)>>,
    /// Every job admitted and not yet ended, run or not.
    pending: Mutex<BTreeSet<u64>>,
    admitted: AtomicU64,
    /// The device worker, which runs the jobs nobody observes.
    worker: Waker,
    /// Woken whenever a job ends.
    waiters: Mutex<Waiters>,
}

type Job = Pin<Box<dyn Future<Output = ()> + Send>>;

impl Transfers {
    pub(crate) fn new(worker: Waker) -> Self {
        Self {
            incoming: Mutex::default(),
            running: Mutex::default(),
            pending: Mutex::default(),
            admitted: AtomicU64::new(0),
            worker,
            waiters: Mutex::default(),
        }
    }

    fn enqueue(&self, job: impl Future<Output = ()> + Send + 'static) {
        let id = self.admitted.fetch_add(1, Ordering::Relaxed);
        self.pending.lock().unwrap().insert(id);
        self.incoming.lock().unwrap().push((id, Box::pin(job)));
        self.worker.wake_by_ref();
    }

    /// Advances every job once; true when none remains. While another thread advances them,
    /// false.
    pub(crate) fn poll(&self) -> bool {
        let Ok(mut running) = self.running.try_lock() else {
            return false;
        };
        running.append(&mut self.incoming.lock().unwrap());
        let mut context = Context::from_waker(&self.worker);
        let mut ended = SmallVec::<[u64; 4]>::new();
        running.retain_mut(|(id, job)| {
            let pending = job.as_mut().poll(&mut context).is_pending();
            if !pending {
                ended.push(*id);
            }
            pending
        });
        let idle = running.is_empty();
        drop(running);
        if !ended.is_empty() {
            let mut pending = self.pending.lock().unwrap();
            ended.iter().for_each(|id| {
                pending.remove(id);
            });
            drop(pending);
            self.waiters.lock().unwrap().take().wake();
        }
        idle && self.incoming.lock().unwrap().is_empty()
    }

    /// Each job admitted so far that has not ended, as [`Self::ended`] takes it.
    pub(crate) fn submitted(&self) -> Vec<u64> {
        self.pending.lock().unwrap().iter().copied().collect()
    }

    /// Whether job `sequence` ended; `waker` is woken at the next end otherwise.
    pub(crate) fn ended(&self, sequence: u64, waker: &Waker) -> bool {
        let ended = || !self.pending.lock().unwrap().contains(&sequence);
        ended() || {
            self.waiters.lock().unwrap().register(waker);
            ended()
        }
    }
}

struct TransferContext {
    driver: Arc<crate::driver::Driver>,
    allocator: Arc<crate::buffer::Allocator>,
    work: super::Work,
}

impl TransferContext {
    fn inline(&self, pairs: &[Pair<'_>]) -> Result<bool> {
        let pieces = self.plan(Direction::Write, pairs)?;
        let mut acquire = std::pin::pin!(self.driver.dma.acquire());
        let std::task::Poll::Ready(result) = acquire.as_mut().poll(&mut Context::from_waker(Waker::noop())) else {
            return Ok(false);
        };
        let mut transfer = result?;
        for piece in &pieces {
            let pair = &pairs[piece.pair];
            let result = transfer
                .access((piece.view, Access::Write))
                .and_then(|()| match pair.pinned {
                    Some(backing) => transfer.access(Region::host(
                        Backing {
                            offset: backing.offset + piece.at,
                            ..backing
                        },
                        piece.view.len(),
                        Access::Read,
                    )),
                    None => Ok(()),
                });
            match result {
                Ok(()) => {}
                Err(Error::Busy) => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        let mut stores = Stores::new();
        for piece in &pieces {
            let pair = &pairs[piece.pair];
            let Way::Window(bytes) = piece.way else {
                unreachable!("small writes use the window");
            };
            stores.push((
                piece,
                bytes,
                pair.pinned
                    .map(|backing| (Arc::clone(backing.dmabuf), piece.at + backing.offset)),
            ));
        }
        self.store(&mut stores)?;
        Ok(true)
    }

    fn topology(&self) -> Topology {
        self.allocator.topology()
    }

    fn acquire(&self, direction: Direction, pairs: &[Pair<'_>]) -> Result<Vec<Lease>> {
        let pieces = self.plan(direction, pairs)?;
        let (host_access, device_access) = match direction {
            Direction::Write => (Access::Read, Access::Write),
            Direction::Read => (Access::Write, Access::Read),
        };
        let admission = Admission::begin();
        let mut leases = Vec::new();
        let mut entries = SLOT;
        for piece in pieces {
            if entries + piece.entries() > SLOT {
                leases.push(Lease::new(SLOT * 2));
                entries = 0;
            }
            entries += piece.entries();
            let pair = &pairs[piece.pair];
            let regions =
                std::iter::once(Region::from((piece.view, device_access))).chain(pair.pinned.map(|backing| {
                    Region::host(
                        Backing {
                            offset: backing.offset + piece.at,
                            ..backing
                        },
                        piece.view.len(),
                        host_access,
                    )
                }));
            leases
                .last_mut()
                .expect("a piece creates its slot")
                .enqueue(&admission, regions)?;
        }
        Ok(leases)
    }

    async fn run(&self, direction: Direction, pairs: &[Pair<'_>], access: Vec<Lease>) -> Result<()> {
        let pieces = self.plan(direction, pairs)?;
        if pieces.is_empty() {
            return Ok(());
        }
        let mut access = access.into_iter();
        let lease = access.next().expect("planned slot");
        self.work.collect(Contention::Wait);
        lease.ready().await?;
        let clusters = Topology {
            chips: 1,
            ..self.topology()
        }
        .clusters();
        let mut transfer = self.driver.dma.acquire().await?;
        transfer.hold(lease);
        let mut entries = 0;
        // Window stores wait until the slot holds every access of its pieces, so a conflict found
        // later stores nothing.
        let mut stores = Stores::new();
        for piece in &pieces {
            let pair = &pairs[piece.pair];
            if entries + piece.entries() > SLOT {
                self.store(&mut stores)?;
                transfer.submit().await?;
                let lease = access.next().expect("planned slot");
                lease.ready().await?;
                transfer = self.driver.dma.acquire().await?;
                transfer.hold(lease);
                entries = 0;
            }
            entries += piece.entries();
            let chip = &self.driver.chips[piece.chip];
            let (size, address) = (piece.view.len(), chip.base + piece.view.addr() as u64);
            match piece.way {
                Way::Window(bytes) => stores.push((
                    piece,
                    bytes,
                    pair.pinned
                        .map(|backing| (Arc::clone(backing.dmabuf), piece.at + backing.offset())),
                )),
                // Stripes take turns among the chip's clusters, so each runs on a channel of its own.
                Way::Stripes(registration, count) => {
                    let backing = pair.pinned.expect("a registration comes with its backing");
                    let window = chip.base - rngd::DEVICE_MEMORY.start;
                    let share = size.div_ceil(count).next_multiple_of(4096).max(1);
                    for (stripe, start) in (0..size).step_by(share).enumerate() {
                        let cluster = piece.chip * clusters + stripe % clusters;
                        let len = share.min(size - start) as u64;
                        transfer.push(Route::Cluster(Copy::new(
                            cluster,
                            direction,
                            Arc::clone(registration),
                            backing.offset() + piece.at + start,
                            window,
                            address + start as u64,
                            len,
                        )));
                    }
                }
                Way::Chunks => {
                    for range in Command::chunks(size) {
                        let host = match pair.pinned {
                            Some(backing) => Host::Owned(
                                Arc::clone(backing.dmabuf),
                                (backing.offset() + piece.at + range.start) as u64,
                            ),
                            None => Host::Borrowed((pair.host.start + piece.at + range.start) as u64),
                        };
                        transfer.push(Route::Channel(Command::new(
                            chip.index,
                            direction,
                            host,
                            address + range.start as u64,
                            range.len() as u64,
                        )));
                    }
                }
            }
        }
        self.store(&mut stores)?;
        // A slot of window stores alone has no command, so its submission only releases it.
        transfer.submit().await
    }

    /// Every chip piece of `pairs` and how it moves. A write of at most [`MMIO_WRITE_LIMIT`]
    /// bytes in all goes through the BAR4 window; memory mapped for this device's clusters moves on
    /// their channels, any other on the kernel driver's.
    fn plan<'a>(&self, direction: Direction, pairs: &'a [Pair<'a>]) -> Result<SmallVec<[Piece<'a>; 8]>> {
        let small = direction == Direction::Write
            && pairs.iter().map(|pair| pair.host.len()).sum::<usize>() <= MMIO_WRITE_LIMIT;
        let mut pieces = SmallVec::new();
        for (index, pair) in pairs.iter().enumerate() {
            let size = pair.view.len();
            let registration = pair.pinned.and_then(|pinned| pinned.mapped_to(&self.allocator));
            for (piece, chip) in self.pieces(&pair.view, pair.host.len())?.enumerate() {
                let at = piece * size;
                let way = match (small, pair.source, registration) {
                    (true, source, _) => Way::Window(source.map(|source| &source[at..at + size])),
                    (_, _, Some(registration)) => Way::Stripes(registration, self.stripes(size)),
                    _ => Way::Chunks,
                };
                pieces.push(Piece {
                    chip,
                    view: pair.view.on(chip)?,
                    at,
                    pair: index,
                    way,
                });
            }
        }
        Ok(pieces)
    }

    /// Stores each of `stores`' bytes through its chip's BAR4 window, then releases its lease.
    fn store(&self, stores: &mut Stores<'_>) -> Result<()> {
        for (piece, bytes, allocation) in stores.drain(..) {
            let mmap;
            let bytes = match bytes {
                Some(bytes) => bytes,
                None => {
                    let (allocation, offset) = allocation.as_ref().expect("pinned window source");
                    // SAFETY: the host read lease excludes DMA writes throughout the store.
                    mmap = unsafe { memmap2::Mmap::map(allocation.fd()) }
                        .map_err(|error| Error::Device(format!("mapping window source: {error}")))?;
                    &mmap[*offset..*offset + piece.view.len()]
                }
            };
            self.driver.chips[piece.chip]
                .window
                .write(piece.view.addr() as u64, bytes)?;
        }
        Ok(())
    }

    /// Copies a chip piece of `size` mapped bytes splits into: one per channel its chip's
    /// clusters own, as long as each share is at least [`STRIPE`].
    fn stripes(&self, size: usize) -> usize {
        (size / STRIPE).clamp(1, crate::driver::chip::FIRMWARE_CHANNELS)
    }

    /// The ranks of the chips `view` selects, once its device and length match `host` bytes.
    fn pieces(&self, view: &BufferSlice<'_>, host: usize) -> Result<impl Iterator<Item = usize> + use<>> {
        if !view.buffer().belongs_to(&self.allocator) {
            return Err(Error::ForeignBuffer);
        }
        if view.kind() == Kind::Sram {
            return Err(Error::Invalid("the host cannot transfer SRAM".into()));
        }
        if view.len().checked_mul(view.chip_mask().count_ones() as usize) != Some(host) {
            return Err(Error::Invalid(format!(
                "{host} host bytes do not match {} bytes on {} chips",
                view.len(),
                view.chip_mask().count_ones()
            )));
        }
        let chips = view.chip_mask();
        Ok((0..usize::from(self.topology().chips)).filter(move |chip| chips & (1 << chip) != 0))
    }
}

/// One pair of a transfer: device bytes and the host bytes they move to or from.
struct Pair<'a> {
    view: BufferSlice<'a>,
    /// The host bytes' addresses.
    host: Range<usize>,
    pinned: Option<Backing<'a>>,
    /// The bytes a write copies, which the BAR4 window can store directly.
    source: Option<&'a [u8]>,
}

impl Pair<'_> {
    /// The pair as a transfer that outlives the caller's borrows holds it; only a pinned pair is
    /// retained.
    fn retain(&self) -> Retained {
        let backing = self.pinned.expect("a retained pair is pinned");
        Retained {
            buffer: self.view.retain(),
            chips: self.view.chip_mask(),
            host: self.host.clone(),
            allocation: Arc::clone(backing.dmabuf),
            registration: backing.registration.cloned(),
            offset: backing.offset(),
        }
    }
}

/// A pinned [`Pair`] that owns what it borrowed: its device bytes, and a reservation that keeps
/// its host bytes for this transfer ahead of any admitted later.
struct Retained {
    buffer: crate::Buffer,
    chips: u8,
    host: Range<usize>,
    allocation: Arc<DmaBuf>,
    registration: Option<Arc<Registration>>,
    offset: usize,
}

impl Retained {
    /// The pair this holds, borrowing from it.
    fn pair(&self) -> Pair<'_> {
        Pair {
            view: (self.buffer.slice(..).expect("a buffer holds its own range")).selecting(self.chips),
            host: self.host.clone(),
            pinned: Some(Backing {
                dmabuf: &self.allocation,
                registration: self.registration.as_ref(),
                offset: self.offset,
            }),
            source: None,
        }
    }
}

/// How a chip piece's bytes move.
#[derive(Clone, Copy)]
enum Way<'a> {
    /// Host stores through the chip's BAR4 window; only small writes take it.
    Window(Option<&'a [u8]>),
    /// Copies the chip's clusters run on their PDMA channels, `count` stripes taking turns
    /// among them.
    Stripes(&'a Arc<Registration>, usize),
    /// The kernel driver's channels.
    Chunks,
}

/// One chip's share of a pair, with how it moves.
struct Piece<'a> {
    chip: usize,
    view: BufferSlice<'a>,
    /// Where this chip's bytes start among the pair's host bytes.
    at: usize,
    pair: usize,
    way: Way<'a>,
}

type Stores<'a> = ArrayVec<(&'a Piece<'a>, Option<&'a [u8]>, Option<(Arc<DmaBuf>, usize)>), SLOT>;

impl Piece<'_> {
    /// The slot entries this piece takes: a window store one, a copy one per stripe or chunk.
    fn entries(&self) -> usize {
        match self.way {
            Way::Window(_) => 1,
            Way::Stripes(_, count) => count,
            Way::Chunks => Command::chunks(self.view.len()).len(),
        }
    }
}

/// The largest write batch that goes through the BAR4 window: below it, a BAR4 store beats a DMA
/// round trip, mapped memory or not.
const MMIO_WRITE_LIMIT: usize = 2 << 10;
/// The smallest share of a chip piece a stripe moves: below it a stripe's own request costs more
/// than running beside the others saves.
const STRIPE: usize = 256 << 10;
/// Entries one transfer holds.
const SLOT: usize = 64;
// A pair takes at most a stripe per channel, of a u8 channel mask, on every chip.
const _: () = assert!(rngd::MAX_CHIPS as usize * u8::BITS as usize <= SLOT);

#[cfg(test)]
mod tests {
    use std::task::{Context, Poll, Waker};

    use super::*;

    /// Dropping an observer during an earlier access does not cancel any transfer pieces.
    #[test]
    fn completes_unobserved_transfer() {
        let device = crate::test::device(&[0, 1], 8);
        let len = MMIO_WRITE_LIMIT + 512;
        let mut host = crate::test::mapped(&device, vec![0, 0], len * 2);
        host.set_registration(None);
        let mut buffer = device.alloc(Kind::Dram, len).unwrap();
        let earlier = crate::buffer::Lease::acquire([(buffer.slice(..).unwrap(), Access::Write)]).unwrap();
        let fake = crate::driver::dma::fixture::transport(&device.driver.dma);
        let mut submitted = Box::pin(device.write([(host.slice(..), buffer.slice_mut(..).unwrap())]));
        assert!(
            submitted
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(submitted);
        drop(earlier);
        device.work().complete(Contention::Wait);
        assert_eq!(fake.pending().len(), 2);
    }

    /// A submitted transfer crosses slot boundaries after its observer is dropped.
    #[test]
    fn completes_every_slot() {
        let device = crate::test::with(&[0], 8, Vec::new(), 1 << 20);
        let len = 4096;
        let host = crate::test::mapped(&device, vec![0], len * 65);
        let mut host = host;
        host.set_registration(None);
        let buffer = device.alloc(Kind::Dram, len * 65).unwrap();
        let earlier = crate::buffer::Lease::acquire([(buffer.slice(..).unwrap(), Access::Write)]).unwrap();
        let fake = crate::driver::dma::fixture::transport(&device.driver.dma);
        let mut submitted = Box::pin(
            device.write(
                host.slice(..)
                    .chunks(len)
                    .zip((0..65).map(|index| BufferSliceMut(buffer.slice(index * len..(index + 1) * len).unwrap()))),
            ),
        );
        assert!(
            submitted
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        let mut synchronized = Box::pin(device.synchronize());
        drop(submitted);
        assert!(
            synchronized
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(earlier);
        let mut count = 0;
        for _ in 0..8 {
            device.work().complete(Contention::Wait);
            for tag in fake.pending() {
                fake.respond(tag, Ok(0));
                count += 1;
            }
        }
        assert_eq!(count, 65);
        assert_eq!(
            synchronized.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(()))
        );
    }

    /// A running job may admit another transfer, as an epilogue reading pinned memory back does,
    /// and the job that admitted it still ends.
    #[test]
    fn admits_transfers_from_a_running_job() {
        let transfers = Arc::new(super::Transfers::new(Waker::noop().clone()));
        let inner = Arc::clone(&transfers);
        transfers.enqueue(async move { inner.enqueue(async {}) });
        let (polled, idle) = std::sync::mpsc::channel();
        let poller = Arc::clone(&transfers);
        std::thread::spawn(move || polled.send((poller.poll(), poller.poll())).unwrap());
        let idle = idle
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("polling a job that admits a transfer returns");
        assert_eq!(idle, (false, true));
        assert!(transfers.submitted().is_empty());
    }

    /// Transfer futures can move to another thread before completion.
    #[test]
    fn transfers_are_send() {
        fn send(_: impl Send) {}
        let device = crate::test::device(&[0], 8);
        let mut buffer = device.alloc(Kind::Dram, 4096).unwrap();
        let mut host = [0; 4096];
        send(device.read([(buffer.slice(..).unwrap(), (&mut host[..]).into())]));
        send(device.write([((&host[..]).into(), buffer.slice_mut(..).unwrap())]));
    }

    /// Without firmware DMA, `map` refuses pinned memory, and the kernel driver's channels move it
    /// instead of a cluster copy.
    #[test]
    fn moves_pinned_memory_on_driver_channels_without_firmware_dma() {
        let mut context = Context::from_waker(Waker::noop());
        let (device, _file) = crate::test::unreserved(&[0], 8);
        let mut host = crate::test::pinned();
        assert_eq!(
            std::pin::pin!(device.map(&mut host)).poll(&mut context),
            Poll::Ready(Err(Error::Unsupported(crate::driver::NO_FIRMWARE_DMA.into())))
        );
        let len = MMIO_WRITE_LIMIT + 512;
        let mut buffer = device.alloc(Kind::Dram, len).unwrap();
        let pairs = [(host.slice(..len), buffer.slice_mut(..).unwrap())];
        assert!(Box::pin(device.write(pairs)).as_mut().poll(&mut context).is_pending());
        assert!(device.driver.mailboxes.lock().unwrap()[0].idle());
    }

    /// A transfer covers the selected chips in rank order and needs an exact host length.
    #[test]
    fn validates_transfer_extents() {
        let device = crate::test::device(&[3, 1], 8);
        let context = TransferContext {
            driver: Arc::clone(&device.driver),
            allocator: Arc::clone(&device.allocator),
            work: device.work(),
        };
        let buffer = device.alloc(Kind::Dram, 16).unwrap();
        for (view, bytes, expected) in [
            (buffer.slice(..).unwrap(), 32, vec![3, 1]),
            (buffer.slice(..).unwrap().on(1).unwrap(), 16, vec![1]),
        ] {
            assert_eq!(
                context
                    .pieces(&view, bytes)
                    .unwrap()
                    .map(|index| device.driver.chips[index].index)
                    .collect::<Vec<_>>(),
                expected,
            );
            assert!(matches!(context.pieces(&view, bytes - 1), Err(Error::Invalid(_))));
        }
        let sram = device.alloc(Kind::Sram, 16).unwrap();
        assert!(matches!(
            context.pieces(&sram.slice(..).unwrap(), 32),
            Err(Error::Invalid(_))
        ));
        let foreign = crate::test::allocator(0..512, 512).alloc(Kind::Dram, 16).unwrap();
        assert!(matches!(
            context.pieces(&foreign.slice(..).unwrap(), 16),
            Err(Error::ForeignBuffer)
        ));
    }
}
