//! Host transfers through queue slots that callers submit and any poller publishes and retires.

use std::iter::Peekable;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Waker};

use arrayvec::ArrayVec;

use crate::arch::rngd;
use crate::device::Device;
use crate::driver::dma::{Command, Direction, Host, Route, Transfer};
use crate::memory::Kind;
use crate::pinned::{Backing, Mapping};
use crate::{Access, BufferSlice, BufferSliceMut, Error, HostBufferSlice, HostBufferSliceMut, Result, Topology};

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

impl Device {
    /// Copies host bytes into exclusive device views once the launches and transfers admitted before
    /// it that use those bytes ended. Selected chips consume host bytes in rank order. Ordinary host
    /// slices finish synchronously after submission, including for forgotten futures.
    pub async fn write(
        &self,
        pairs: impl IntoIterator<Item = (HostBufferSlice<'_>, BufferSliceMut<'_>)>,
    ) -> Result<()> {
        let mut pairs = pairs.into_iter().peekable();
        loop {
            let batch = self.batch(&mut pairs, |(host, view)| (view.as_slice(), host.backing));
            if batch.is_empty() {
                return Ok(());
            }
            let transfer = self
                .prepare(
                    Direction::Write,
                    batch
                        .iter()
                        .map(|(host, view)| {
                            let range = host.bytes.as_ptr_range();
                            (view.as_slice(), range.start.addr()..range.end.addr(), host.backing)
                        })
                        .collect(),
                )
                .await?;
            if batch.iter().map(|(host, _)| host.bytes.len()).sum::<usize>() > MMIO_WRITE_LIMIT {
                transfer.submit().await?;
                continue;
            }
            for (host, view) in &batch {
                let view = view.as_slice();
                for (chip, bytes) in view.chips().zip(host.bytes.chunks_exact(view.len().max(1))) {
                    self.driver.chips[chip.chips.trailing_zeros() as usize]
                        .window
                        .write(chip.addr() as u64, bytes)?;
                }
            }
        }
    }

    /// Copies shared device views into exclusive host views once the launches and transfers admitted
    /// before it that write those views ended; byte lengths must match exactly.
    pub async fn read(&self, pairs: impl IntoIterator<Item = (BufferSlice<'_>, HostBufferSliceMut<'_>)>) -> Result<()> {
        let mut pairs = pairs.into_iter().peekable();
        loop {
            let batch = self.batch(&mut pairs, |(view, host)| (*view, host.backing));
            if batch.is_empty() {
                return Ok(());
            }
            self.prepare(
                Direction::Read,
                batch
                    .into_iter()
                    .map(|(view, host)| {
                        let range = host.bytes.as_ptr_range();
                        (view, range.start.addr()..range.end.addr(), host.backing)
                    })
                    .collect(),
            )
            .await?
            .submit()
            .await?;
        }
    }

    /// The pairs one transfer holds, from the front of `pairs`: every chip piece of a pair takes an
    /// entry per stripe.
    fn batch<'p, P>(
        &self,
        pairs: &mut Peekable<impl Iterator<Item = P>>,
        side: impl Fn(&P) -> (BufferSlice<'_>, Option<Backing<'p>>),
    ) -> ArrayVec<P, SLOT> {
        let mut room = SLOT;
        let mut fits = |pair: &P| {
            let (view, backing) = side(pair);
            let stripes = self.mapping(backing).map_or(1, |_| self.stripes(view.len()));
            room.checked_sub(view.chips.count_ones() as usize * stripes)
                .map(|left| room = left)
                .is_some()
        };
        std::iter::from_fn(|| pairs.next_if(&mut fits)).collect()
    }

    /// The mapping through which this device's clusters reach `backing`, if it has one.
    fn mapping<'b>(&self, backing: Option<Backing<'b>>) -> Option<&'b Arc<Mapping>> {
        backing
            .and_then(|backing| backing.mapping)
            .filter(|mapping| mapping.table.belongs_to(&self.allocations))
    }

    /// Copies a chip piece of `size` mapped bytes splits into: one per channel its chip's
    /// clusters own, as long as each share is at least [`STRIPE`].
    fn stripes(&self, size: usize) -> usize {
        let clusters = rngd::clusters(Topology {
            chips: 1,
            ..self.topology()
        });
        let channels = (0..clusters).map(|cluster| rngd::channels(cluster, clusters).count_ones() as usize);
        (size / STRIPE).clamp(1, channels.sum())
    }

    fn pieces(&self, view: &BufferSlice<'_>, host: usize) -> Result<impl Iterator<Item = usize> + use<>> {
        if !view.buffer.belongs_to(&self.allocations) {
            return Err(Error::ForeignBuffer);
        }
        if view.kind() == Kind::Sram {
            return Err(Error::Invalid("the host cannot transfer SRAM".into()));
        }
        if view.len().checked_mul(view.chips.count_ones() as usize) != Some(host) {
            return Err(Error::Invalid(format!(
                "{host} host bytes do not match {} bytes on {} chips",
                view.len(),
                view.chips.count_ones()
            )));
        }
        let chips = view.chips;
        Ok((0..usize::from(self.topology().chips)).filter(move |chip| chips & (1 << chip) != 0))
    }

    /// A transfer of `pairs` after the work admitted before it on their device views ended. It holds
    /// no memory while it waits, so two transfers never wait for each other.
    async fn prepare(
        &self,
        direction: Direction,
        pairs: ArrayVec<(BufferSlice<'_>, Range<usize>, Option<Backing<'_>>), SLOT>,
    ) -> Result<Transfer<'_>> {
        let (host_access, device_access) = match direction {
            Direction::Write => (Access::Read, Access::Write),
            Direction::Read => (Access::Write, Access::Read),
        };
        let mut waits = pairs
            .iter()
            .map(|(view, ..)| view.ready(device_access))
            .collect::<ArrayVec<_, SLOT>>();
        let waiting = {
            let mut context = Context::from_waker(Waker::noop());
            waits
                .iter_mut()
                .any(|wait| Pin::new(wait).poll(&mut context).is_pending())
        };
        if waiting {
            // Work the device finished but nobody retired yet ends now, not on the worker's next turn.
            self.worker.launches.retire();
            self.driver.transfers.progress();
        }
        for wait in waits {
            wait.await?;
        }
        let mut transfer = self.driver.transfers.acquire().await?;
        let mut entries = 0;
        // The pinned allocations the slot leases. The host views of one call are disjoint, so a
        // later pair in one of them retains its lease instead of conflicting with it. A pair that
        // finds the slot full is not recorded: it leases for itself, and the split that follows
        // starts the record over with it.
        let mut leased = arrayvec::ArrayVec::<Backing<'_>, SLOT>::new();
        let clusters = rngd::clusters(Topology {
            chips: 1,
            ..self.topology()
        });
        for (view, bytes, backing) in pairs {
            let size = view.len();
            // Each slot keeps the host lease until every command answers, observed or not.
            let mut lease = match backing {
                Some(backing) if !leased.iter().any(|other| other.shares(&backing)) => {
                    let _ = leased.try_push(backing);
                    Some(self.admit(|| backing.lease(host_access))?)
                }
                _ => None,
            };
            // Memory mapped for this device's clusters moves on their channels, any other on the driver's.
            let mapping = self.mapping(backing);
            for (piece, index) in self.pieces(&view, bytes.len())?.enumerate() {
                let count = mapping.map_or_else(|| Command::chunks(size).len(), |_| self.stripes(size));
                if entries + count > SLOT {
                    transfer.submit().await?;
                    transfer = self.driver.transfers.acquire().await?;
                    leased.clear();
                    if let Some(backing) = backing {
                        let _ = leased.try_push(backing);
                        if lease.is_none() {
                            lease = Some(self.admit(|| backing.lease(host_access))?);
                        }
                    }
                    entries = 0;
                }
                entries += count;
                let chip = &self.driver.chips[index];
                let (at, address) = (piece * size, chip.base + view.addr() as u64);
                let piece = view.on(index)?;
                self.admit(|| transfer.access(piece, device_access))?;
                match (&backing, mapping) {
                    // Stripes take turns among the chip's clusters, so each runs on a channel of its own.
                    (Some(backing), Some(mapping)) => {
                        let window = chip.base - rngd::DEVICE_MEMORY.start;
                        let share = size.div_ceil(self.stripes(size)).next_multiple_of(4096).max(1);
                        for (stripe, start) in (0..size).step_by(share).enumerate() {
                            let cluster = index * clusters + stripe % clusters;
                            let host = mapping.operand(window, cluster, (backing.offset + at + start) as u32);
                            let len = share.min(size - start) as u64;
                            transfer.push(Route::Cluster {
                                cluster,
                                segment: direction.segment(host, address + start as u64, len),
                                _mapping: Arc::clone(mapping),
                                _host: lease.take().unwrap_or_else(|| backing.retain()),
                                id: None,
                            });
                        }
                    }
                    (backing, _) => {
                        for range in Command::chunks(size) {
                            let host = match backing {
                                Some(backing) => Host::Owned(
                                    lease.take().unwrap_or_else(|| backing.retain()),
                                    (backing.offset + at + range.start) as u64,
                                ),
                                None => Host::Borrowed((bytes.start + at + range.start) as u64),
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
        }
        Ok(transfer)
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::FileExt;
    use std::task::{Context, Poll, Waker};

    use furiosa_opt_ipc::{Request, Response, wire};

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

    /// One read lands pieces of one pinned allocation, as a multi-chip view read chip by chip into
    /// one host range does; the read holds the allocation as a writer until every piece answers.
    #[test]
    fn reads_pieces_into_one_pinned_allocation() {
        let device = crate::test::device(&[0, 1], 8);
        let buffer = device.alloc(Kind::Dram, 1024).unwrap();
        let mut host = crate::test::pinned();
        let host = host.slice_mut(..2048);
        let backing = host.backing.unwrap();
        let pairs = buffer.slice(..).unwrap().chips().zip(host.chunks(1024));
        let mut read = Box::pin(device.read(pairs));
        let published = read.as_mut().poll(&mut Context::from_waker(Waker::noop()));
        assert!(published.is_pending(), "{published:?}");
        assert!(matches!(backing.lease(Access::Read), Err(Error::Busy)));
    }

    /// Cancelled large reads keep host writers exclusive until the driver answers.
    #[test]
    fn retains_writer_when_cancelled() {
        let size = 48_usize << 30;
        let mut device = crate::test::device(&[0, 1, 2, 3, 4, 5, 6, 7], 8);
        device.allocations = Arc::new(crate::buffer::Allocations::new(
            device.topology(),
            crate::buffer::Ranges::new(0..size, size),
        ));
        let buffer = device.alloc(Kind::Dram, size).unwrap();
        let file = tempfile::tempfile().unwrap();
        file.set_len((size * 8) as u64).unwrap();
        let mut host = crate::Pinned::new(file.into(), size * 8).unwrap();
        let host = host.slice_mut(..);
        let backing = host.backing.unwrap();
        let bytes = host.bytes.as_ptr_range();
        let mut transfer = Box::pin(
            device.prepare(
                Direction::Read,
                [(
                    buffer.slice(..).unwrap(),
                    bytes.start.addr()..bytes.end.addr(),
                    Some(backing),
                )]
                .into_iter()
                .collect(),
            ),
        );
        assert!(
            transfer
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(transfer);
        assert!(matches!(backing.lease(Access::Read), Err(Error::Busy)));
    }

    /// Mapped pinned memory moves as a copy its chip's cluster runs, between the host operand and
    /// the device bytes, which a later transfer waits for until the cluster answers, observed or not.
    #[test]
    fn copies_mapped_memory_on_clusters() {
        let mut context = Context::from_waker(Waker::noop());
        for write in [true, false] {
            let (device, file) = crate::test::running(&[0], 8);
            let mut host = crate::test::mapped(&device, vec![0, 256], 4096);
            let table = host.mapping.as_ref().unwrap().table.addr() as u64;
            let operand = rngd::dma::Operand::new(rngd::DEVICE_MEMORY.start + table, 0)
                .unwrap()
                .encode();
            // Past the bytes a write takes through BAR4 instead.
            let len = MMIO_WRITE_LIMIT + 512;
            let mut buffer = device.alloc(Kind::Dram, len).unwrap();
            let address = rngd::DEVICE_MEMORY.start + buffer.addr() as u64;
            let published = if write {
                let pairs = [(host.slice(..len), buffer.slice_mut(..).unwrap())];
                Box::pin(device.write(pairs)).as_mut().poll(&mut context)
            } else {
                let pairs = [(buffer.slice(..).unwrap(), host.slice_mut(..len))];
                Box::pin(device.read(pairs)).as_mut().poll(&mut context)
            };
            assert!(published.is_pending());
            let mut frame = [0_u64; 8];
            file.read_exact_at(bytemuck::cast_slice_mut(&mut frame), 4096 + 8)
                .unwrap();
            let (source, destination) = if write { (operand, address) } else { (address, operand) };
            let copy = Request::Copy {
                source,
                destination,
                len: len as u64,
            };
            assert_eq!(wire::decode::<Request<'_>>(&frame), Ok(copy));
            let mut probe = || {
                let bytes = [1; 16];
                std::pin::pin!(device.write([(bytes[..].into(), buffer.slice_mut(..16).unwrap())])).poll(&mut context)
            };
            assert_eq!(probe(), Poll::Pending);
            crate::test::respond(&file, &[(1, Response::Copy)]);
            device.driver.transfers.progress();
            assert_eq!(probe(), Poll::Ready(Ok(())));
        }
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
            Poll::Ready(Err(Error::Device(crate::driver::NO_FIRMWARE_DMA.into())))
        );
        let len = MMIO_WRITE_LIMIT + 512;
        let mut buffer = device.alloc(Kind::Dram, len).unwrap();
        let pairs = [(host.slice(..len), buffer.slice_mut(..).unwrap())];
        assert!(Box::pin(device.write(pairs)).as_mut().poll(&mut context).is_pending());
        assert!(device.driver.links.lock().unwrap()[0].idle());
    }

    /// Mapped memory of several shares moves in page-aligned stripes, which take turns among the
    /// chip's clusters and together cover the piece.
    #[test]
    fn stripes_mapped_memory_across_clusters() {
        let len = 4 * STRIPE + 4096;
        let (device, files) = crate::test::clusters(8, 2 * len);
        let host = crate::test::mapped(&device, vec![0, 256], len);
        let table = rngd::DEVICE_MEMORY.start + host.mapping.as_ref().unwrap().table.addr() as u64;
        let mut buffer = device.alloc(Kind::Dram, len).unwrap();
        let address = rngd::DEVICE_MEMORY.start + buffer.addr() as u64;
        let pairs = [(host.slice(..), buffer.slice_mut(..).unwrap())];
        let published = Box::pin(device.write(pairs))
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()));
        assert!(published.is_pending());
        let share = STRIPE + 4096;
        for (cluster, file) in files.iter().enumerate() {
            let mut ring = [0_u64; 16];
            file.read_exact_at(bytemuck::cast_slice_mut(&mut ring), 4096).unwrap();
            let mut frames = ring.split(|&word| furiosa_opt_abi::ring::header::unpack(word).is_some());
            frames.next();
            let copies = frames
                .take(2)
                .map(|frame| wire::decode::<Request<'_>>(frame).unwrap())
                .collect::<Vec<_>>();
            let expected = [cluster, cluster + 2].map(|stripe| {
                let start = stripe * share;
                Request::Copy {
                    source: rngd::dma::Operand::new(table + cluster as u64 * 256, start as u32)
                        .unwrap()
                        .encode(),
                    destination: address + start as u64,
                    len: share.min(len - start) as u64,
                }
            });
            assert_eq!(copies, expected);
        }
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
