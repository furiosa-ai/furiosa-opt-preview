//! Host transfers through queue slots that callers submit and any poller publishes and completes.

use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Waker};

use arrayvec::ArrayVec;
use smallvec::SmallVec;

use crate::arch::rngd;
use crate::device::{Contention, Device};
use crate::driver::dma::{Command, Copy, Direction, Host, Route};
use crate::memory::Kind;
use crate::pinned::{Backing, PinnedLease, Registration};
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
                let range = host.bytes.as_ptr_range();
                Pair {
                    view: view.0,
                    host: range.start.addr()..range.end.addr(),
                    backing: host.backing,
                    source: Some(host.bytes),
                }
            })
            .collect::<SmallVec<[_; 4]>>();
        self.run(Direction::Write, &pairs).await
    }

    /// Copies shared device views into exclusive host views once the launches and transfers admitted
    /// before it that write those views ended; byte lengths must match exactly.
    pub async fn read(&self, pairs: impl IntoIterator<Item = (BufferSlice<'_>, HostBufferSliceMut<'_>)>) -> Result<()> {
        let pairs = (pairs.into_iter())
            .map(|(view, host)| {
                let range = host.bytes.as_ptr_range();
                Pair {
                    view,
                    host: range.start.addr()..range.end.addr(),
                    backing: host.backing,
                    source: None,
                }
            })
            .collect::<SmallVec<[_; 4]>>();
        self.run(Direction::Read, &pairs).await
    }

    /// Moves `pairs` after the work admitted before it on their device views ended, filling queue
    /// slots in order and submitting each as it fills. It holds no memory while it waits, so two
    /// transfers never wait for each other.
    async fn run(&self, direction: Direction, pairs: &[Pair<'_>]) -> Result<()> {
        let pieces = self.plan(direction, pairs)?;
        if pieces.is_empty() {
            return Ok(());
        }
        let (host_access, device_access) = match direction {
            Direction::Write => (Access::Read, Access::Write),
            Direction::Read => (Access::Write, Access::Read),
        };
        let mut waits = pairs
            .iter()
            .map(|pair| pair.view.ready(device_access))
            .collect::<SmallVec<[_; 4]>>();
        let waiting = {
            let mut context = Context::from_waker(Waker::noop());
            waits
                .iter_mut()
                .any(|wait| Pin::new(wait).poll(&mut context).is_pending())
        };
        if waiting {
            // Work the device finished but nobody collected yet ends now, not on the worker's next turn.
            self.work().complete(Contention::Wait);
        }
        for wait in waits {
            wait.await?;
        }
        let clusters = Topology {
            chips: 1,
            ..self.topology()
        }
        .clusters();
        let mut transfer = self.driver.transfers.acquire().await?;
        let mut entries = 0;
        // The pinned allocations the slot leases; the host views of one call are disjoint, so a
        // later piece in one of them retains the slot's lease instead of conflicting with it.
        let mut leased = ArrayVec::<Backing<'_>, SLOT>::new();
        // Window stores wait until the slot holds every access of its pieces, so a conflict found
        // later stores nothing.
        let mut stores = ArrayVec::<(&Piece<'_>, &[u8], Option<PinnedLease>), SLOT>::new();
        for piece in &pieces {
            let pair = &pairs[piece.pair];
            if entries + piece.entries() > SLOT {
                self.store(&mut stores)?;
                transfer.submit().await?;
                transfer = self.driver.transfers.acquire().await?;
                entries = 0;
                leased.clear();
            }
            entries += piece.entries();
            // The slot keeps each host lease until every command answers, observed or not.
            let mut lease = match pair.backing {
                Some(backing) if !leased.iter().any(|other| other.shares(&backing)) => {
                    let _ = leased.try_push(backing);
                    Some(self.admit(|| backing.lease(host_access))?)
                }
                _ => None,
            };
            self.admit(|| transfer.access(piece.view, device_access))?;
            let chip = &self.driver.chips[piece.chip];
            let (size, address) = (piece.view.len(), chip.base + piece.view.addr() as u64);
            match piece.way {
                Way::Window(bytes) => stores.push((piece, bytes, lease)),
                // Stripes take turns among the chip's clusters, so each runs on a channel of its own.
                Way::Stripes(registration, count) => {
                    let backing = pair.backing.expect("a registration comes with its backing");
                    let window = chip.base - rngd::DEVICE_MEMORY.start;
                    let share = size.div_ceil(count).next_multiple_of(4096).max(1);
                    for (stripe, start) in (0..size).step_by(share).enumerate() {
                        let cluster = piece.chip * clusters + stripe % clusters;
                        let len = share.min(size - start) as u64;
                        transfer.push(Route::Cluster(Copy::new(
                            cluster,
                            direction,
                            Arc::clone(registration),
                            lease
                                .take()
                                .unwrap_or_else(|| backing.retain())
                                .at(backing.offset() + piece.at + start),
                            window,
                            address + start as u64,
                            len,
                        )));
                    }
                }
                Way::Chunks => {
                    for range in Command::chunks(size) {
                        let host = match pair.backing {
                            Some(backing) => Host::Owned(
                                lease.take().unwrap_or_else(|| backing.retain()),
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
            let registration = self.registration(pair.backing);
            for (piece, chip) in self.pieces(&pair.view, pair.host.len())?.enumerate() {
                let at = piece * size;
                let way = match (small, pair.source, registration) {
                    (true, Some(source), _) => Way::Window(&source[at..at + size]),
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
    fn store(&self, stores: &mut ArrayVec<(&Piece<'_>, &[u8], Option<PinnedLease>), SLOT>) -> Result<()> {
        for (piece, bytes, _lease) in stores.drain(..) {
            self.driver.chips[piece.chip]
                .window
                .write(piece.view.addr() as u64, bytes)?;
        }
        Ok(())
    }

    /// The registration through which this device's clusters reach `backing`, if it has one.
    fn registration<'b>(&self, backing: Option<Backing<'b>>) -> Option<&'b Arc<Registration>> {
        backing
            .and_then(|backing| backing.registration())
            .filter(|registration| registration.buffer.belongs_to(&self.allocations))
    }

    /// Copies a chip piece of `size` mapped bytes splits into: one per channel its chip's
    /// clusters own, as long as each share is at least [`STRIPE`].
    fn stripes(&self, size: usize) -> usize {
        (size / STRIPE).clamp(1, crate::driver::chip::FIRMWARE_CHANNELS)
    }

    /// The ranks of the chips `view` selects, once its device and length match `host` bytes.
    fn pieces(&self, view: &BufferSlice<'_>, host: usize) -> Result<impl Iterator<Item = usize> + use<>> {
        if !view.buffer().belongs_to(&self.allocations) {
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
    backing: Option<Backing<'a>>,
    /// The bytes a write copies, which the BAR4 window can store directly.
    source: Option<&'a [u8]>,
}

/// How a chip piece's bytes move.
#[derive(Clone, Copy)]
enum Way<'a> {
    /// Host stores through the chip's BAR4 window; only small writes take it.
    Window(&'a [u8]),
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
        let buffer = device.alloc(Kind::Dram, 16).unwrap();
        for (view, bytes, expected) in [
            (buffer.slice(..).unwrap(), 32, vec![3, 1]),
            (buffer.slice(..).unwrap().on(1).unwrap(), 16, vec![1]),
        ] {
            assert_eq!(
                device
                    .pieces(&view, bytes)
                    .unwrap()
                    .map(|index| device.driver.chips[index].index)
                    .collect::<Vec<_>>(),
                expected,
            );
            assert!(matches!(device.pieces(&view, bytes - 1), Err(Error::Invalid(_))));
        }
        let sram = device.alloc(Kind::Sram, 16).unwrap();
        assert!(matches!(
            device.pieces(&sram.slice(..).unwrap(), 32),
            Err(Error::Invalid(_))
        ));
        let foreign = crate::test::allocations(0..512, 512).alloc(Kind::Dram, 16).unwrap();
        assert!(matches!(
            device.pieces(&foreign.slice(..).unwrap(), 16),
            Err(Error::ForeignBuffer)
        ));
    }
}
