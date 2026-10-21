//! Which admitted work holds each allocation's bytes, until the hardware is done with them.
//!
//! Records are preallocated per launch or transfer slot and linked into their allocation's list in
//! place, so admitting work allocates nothing. Each record carries its admission number, so a wait
//! for the work admitted before it can skip later admissions.

use std::cell::{Cell, UnsafeCell};
use std::ops::Range;
use std::ptr;
use std::sync::{Arc, PoisonError};
use std::task::{Poll, Waker};

use super::{Access, Allocation, BufferSlice};
use crate::memory::Kind;
use crate::{Error, Result};

/// Accesses held until dropped, as a launch or transfer holds its memory until the device
/// finishes: a read excludes overlapping writes on a shared chip, a write every overlapping access.
pub struct Lease {
    records: Box<[Record]>,
    len: usize,
    owner: Owner,
}

impl Lease {
    /// Holds `regions` with each one's access; [`Error::Busy`] when two of them or another lease
    /// conflict, and then this one holds nothing.
    pub fn acquire<'a>(regions: impl IntoIterator<Item = (BufferSlice<'a>, Access)>) -> Result<Self> {
        let regions: Vec<_> = regions.into_iter().collect();
        let mut lease = Self::new(regions.len());
        lease.extend(regions)?;
        Ok(lease)
    }

    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            records: (0..capacity)
                .map(|_| Record {
                    region: UnsafeCell::new(None),
                    prev: Cell::new(ptr::null()),
                    next: Cell::new(ptr::null()),
                })
                .collect(),
            len: 0,
            owner: Owner::Unordered,
        }
    }

    /// Makes `owner` the owner of the regions this batch acquires next.
    pub(crate) fn set_owner(&mut self, owner: Owner) {
        self.owner = owner;
    }

    /// Adds regions to this batch; a conflict releases only the regions this call added, since
    /// the earlier ones guard work already queued.
    pub(crate) fn extend<'a>(&mut self, regions: impl IntoIterator<Item = (BufferSlice<'a>, Access)>) -> Result<()> {
        if let Owner::Ordered(_) = self.owner {
            assert_eq!(self.len, 0, "an ordered owner admits one complete batch");
        }
        let mark = self.len;
        coalesce(regions)
            .try_for_each(|request| self.hold(request))
            .inspect_err(|_| self.truncate(mark))?;
        self.seal();
        Ok(())
    }

    /// A failed operation can leave its allocation unsafe even after its records are released.
    pub(crate) fn poison(&self) {
        for record in &self.records[..self.len] {
            // SAFETY: shared Lease methods preserve region; other owners may change only its links.
            let region = unsafe { &*record.region.get() }.as_ref().expect("owned access");
            let mut active = region.allocation.active.lock().unwrap_or_else(PoisonError::into_inner);
            active.poisoned = true;
            let waiters = active.waiters.take();
            drop(active);
            waiters.wake();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.truncate(0);
    }

    pub(crate) fn capacity(&self) -> usize {
        self.records.len()
    }

    /// Holds `request` in this batch's next record, unless work already holding its bytes conflicts.
    fn hold(&mut self, request: Request<'_>) -> Result<()> {
        // Nothing panics while a list is half updated, so a poisoned lock still guards a whole list.
        let mut active = (request.allocation.active.lock()).unwrap_or_else(PoisonError::into_inner);
        if active.poisoned {
            return Err(Error::Poisoned);
        }
        if request.bytes.is_empty() || request.chips == 0 {
            return Ok(());
        }
        let record = self.records.get(self.len).ok_or(Error::Capacity)?;
        active.admits(&request, self.owner.on(request.allocation))?;
        let sequence = active.admitted;
        active.insert(
            record,
            Region {
                allocation: Arc::clone(request.allocation),
                bytes: request.bytes,
                chips: request.chips,
                access: request.access,
                // Batch members stay non-exempt until intra-launch aliases have been checked.
                owner: Cell::new(Owner::Unordered),
                sequence,
            },
        );
        self.len += 1;
        Ok(())
    }

    /// Gives an ordered batch's records its owner once its own aliases have been checked, which
    /// exempts them from conflicts with other ordered owners.
    fn seal(&self) {
        if let Owner::Unordered = self.owner {
            return;
        }
        for record in &self.records[..self.len] {
            // SAFETY: this Lease owns the region; its allocation mutex excludes list readers.
            let region = unsafe { &*record.region.get() }.as_ref().expect("owned access");
            let mut active = region.allocation.active.lock().unwrap_or_else(PoisonError::into_inner);
            let owner = self.owner.on(&region.allocation);
            region.owner.set(owner);
            if let Owner::Ordered(_) = owner {
                active.unordered -= 1;
            }
        }
    }

    fn truncate(&mut self, len: usize) {
        for record in self.records[len..self.len].iter().rev() {
            // SAFETY: Lease owns this region until unlink; no shared Lease method can run concurrently.
            let region = unsafe { &*record.region.get() }.as_ref().expect("owned access");
            let waiters = {
                let mut active = region.allocation.active.lock().unwrap_or_else(PoisonError::into_inner);
                let (prev, next) = (record.prev.replace(ptr::null()), record.next.replace(ptr::null()));
                // SAFETY: every neighbor remains linked and alive under this allocation's mutex.
                match unsafe { prev.as_ref() } {
                    Some(prev) => prev.next.set(next),
                    None => active.head = next,
                }
                if let Some(next) = unsafe { next.as_ref() } {
                    next.prev.set(prev);
                }
                active.writers -= usize::from(region.access == Access::Write);
                match region.owner.get() {
                    Owner::Unordered => active.unordered -= 1,
                    Owner::Ordered(_) => {}
                }
                if active.head.is_null() {
                    active.bytes = 0..0;
                }
                active.waiters.take()
            };
            // SAFETY: unlink excludes every traversal before clearing the retained allocation reference.
            unsafe { (*record.region.get()).take() };
            waiters.wake();
        }
        self.len = len;
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.clear();
    }
}

impl BufferSlice<'_> {
    /// Resolves once the launches and transfers admitted before this call that conflict with an
    /// `access` to this view have ended: a read waits for writers, a write for every access. Work
    /// admitted later is not waited for. [`Error::Poisoned`] once failed device work left the
    /// memory unknown.
    pub fn ready(self, access: Access) -> impl Future<Output = Result<()>> + Send + use<> {
        let allocation = Arc::clone(&self.buffer.allocation);
        let start = self.buffer.offset + self.offset;
        let (bytes, chips) = (start..start + self.len, self.chips);
        let before = allocation
            .active
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .admitted;
        std::future::poll_fn(move |context| {
            let mut active = allocation.active.lock().unwrap_or_else(PoisonError::into_inner);
            if active.poisoned {
                return Poll::Ready(Err(Error::Poisoned));
            }
            let blocked = (access == Access::Write || active.writers != 0)
                && bytes.start < active.bytes.end
                && active.bytes.start < bytes.end
                && active
                    .regions()
                    .any(|region| region.sequence < before && region.conflicts(&bytes, chips, access));
            if !blocked {
                return Poll::Ready(Ok(()));
            }
            active.waiters.register(context.waker());
            Poll::Pending
        })
    }
}

impl super::Buffer {
    /// [`BufferSlice::ready`] for every byte on every chip.
    pub fn ready(&self, access: Access) -> impl Future<Output = Result<()>> + Send + use<> {
        self.slice(..).expect("a buffer holds its own range").ready(access)
    }
}

/// The wakers of the tasks waiting for some state to change, woken together once it does.
#[derive(Default)]
pub(crate) struct Waiters(Vec<Waker>);

impl Waiters {
    /// Adds `waker` unless one waking the same task already waits.
    pub(crate) fn register(&mut self, waker: &Waker) {
        if !self.0.iter().any(|waiting| waiting.will_wake(waker)) {
            self.0.push(waker.clone());
        }
    }

    /// Every waiting waker, to wake once the lock guarding this list is released.
    pub(crate) fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    pub(crate) fn wake(self) {
        self.0.into_iter().for_each(Waker::wake);
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Owner {
    Ordered(u64),
    #[default]
    Unordered,
}

impl Owner {
    fn on(self, allocation: &Allocation) -> Self {
        if allocation.kind == Kind::DramShared && allocation.allocations.topology().chips > 1 {
            Self::Unordered
        } else {
            self
        }
    }
}

#[derive(Default)]
pub(super) struct Active {
    head: *const Record,
    /// A conservative envelope; removing a record need not shrink it.
    bytes: Range<usize>,
    writers: usize,
    unordered: usize,
    unordered_bytes: Range<usize>,
    poisoned: bool,
    /// Records admitted so far, which numbers the next one.
    admitted: u64,
    /// Woken when a record leaves or the allocation is poisoned.
    waiters: Waiters,
}

// SAFETY: the allocation mutex serializes every list access; linked records remain alive and fixed.
unsafe impl Send for Active {}

impl Active {
    /// Whether `owner` may hold `request`'s bytes beside the work already holding them.
    fn admits(&self, request: &Request<'_>, owner: Owner) -> Result<()> {
        let overlaps = match owner {
            Owner::Ordered(_) => self.unordered != 0 && overlap(&request.bytes, &self.unordered_bytes),
            Owner::Unordered => overlap(&request.bytes, &self.bytes),
        };
        let busy = (request.access == Access::Write || self.writers != 0)
            && overlaps
            && self.regions().any(|region| {
                let exempt = match (owner, region.owner.get()) {
                    (Owner::Ordered(left), Owner::Ordered(right)) => left != right,
                    (Owner::Unordered, _) | (_, Owner::Unordered) => false,
                };
                !exempt && region.conflicts(&request.bytes, request.chips, request.access)
            });
        if busy { Err(Error::Busy) } else { Ok(()) }
    }

    /// Links `record`, now holding `region`, at the head of the list and widens the envelopes over
    /// it.
    fn insert(&mut self, record: &Record, region: Region) {
        self.bytes = if self.head.is_null() {
            region.bytes.clone()
        } else {
            union(&self.bytes, &region.bytes)
        };
        self.unordered_bytes = if self.unordered == 0 {
            region.bytes.clone()
        } else {
            union(&self.unordered_bytes, &region.bytes)
        };
        self.writers += usize::from(region.access == Access::Write);
        self.unordered += 1;
        // SAFETY: this unused record is exclusively owned; publication uses its allocation lock.
        unsafe {
            *record.region.get() = Some(region);
            if let Some(head) = self.head.as_ref() {
                head.prev.set(ptr::from_ref(record));
            }
        }
        record.next.set(self.head);
        self.head = ptr::from_ref(record);
        self.admitted += 1;
    }

    /// The linked records' regions, newest first.
    fn regions(&self) -> impl Iterator<Item = &Region> {
        let mut next = self.head;
        std::iter::from_fn(move || {
            // SAFETY: the allocation lock `self` is borrowed from keeps each linked record and region alive.
            let record = unsafe { next.as_ref() }?;
            next = record.next.get();
            Some(unsafe { &*record.region.get() }.as_ref().expect("linked access"))
        })
    }
}

/// Unused records are unlinked: both neighbors are null.
struct Record {
    region: UnsafeCell<Option<Region>>,
    prev: Cell<*const Record>,
    next: Cell<*const Record>,
}

// SAFETY: records are published only in a boxed slice; moving Lease never moves a linked record.
unsafe impl Send for Record {}
// SAFETY: list access uses the allocation mutex. Only its exclusive Lease owner changes region;
// readers of that region finish under the same mutex before it is unlinked and cleared.
unsafe impl Sync for Record {}

struct Region {
    allocation: Arc<Allocation>,
    bytes: Range<usize>,
    chips: u8,
    access: Access,
    owner: Cell<Owner>,
    /// Records of the allocation admitted before this one.
    sequence: u64,
}

impl Region {
    /// Whether an `access` to `bytes` on `chips` races with this region's.
    fn conflicts(&self, bytes: &Range<usize>, chips: u8, access: Access) -> bool {
        (access == Access::Write || self.access == Access::Write)
            && chips & self.chips != 0
            && overlap(bytes, &self.bytes)
    }
}

/// One contiguous access a batch asks for.
struct Request<'a> {
    allocation: &'a Arc<Allocation>,
    bytes: Range<usize>,
    chips: u8,
    access: Access,
}

/// `regions` as requests: views of one allocation that follow each other on the same chips with
/// the same access, as a slice of blocks binds them, become one.
fn coalesce<'a>(regions: impl IntoIterator<Item = (BufferSlice<'a>, Access)>) -> impl Iterator<Item = Request<'a>> {
    let mut regions = regions.into_iter().peekable();
    std::iter::from_fn(move || {
        let (view, access) = regions.next()?;
        let start = view.buffer.offset + view.offset;
        let mut request = Request {
            allocation: &view.buffer.allocation,
            bytes: start..start + view.len,
            chips: view.chips,
            access,
        };
        while let Some((next, _)) = regions.next_if(|(next, next_access)| {
            Arc::ptr_eq(&next.buffer.allocation, request.allocation)
                && next.chips == request.chips
                && *next_access == access
                && next.buffer.offset + next.offset == request.bytes.end
        }) {
            request.bytes.end += next.len;
        }
        Some(request)
    })
}

fn overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn union(left: &Range<usize>, right: &Range<usize>) -> Range<usize> {
    left.start.min(right.start)..left.end.max(right.end)
}

#[cfg(test)]
mod tests {
    use Access::{Read, Write};

    use super::*;
    use crate::buffer::Buffer;
    use crate::memory::Kind;
    use crate::test::{Wakes, poll, woken};

    fn view(buffer: &Buffer, bytes: Range<usize>, chips: u8) -> BufferSlice<'_> {
        BufferSlice {
            chips,
            ..buffer.slice(bytes).unwrap()
        }
    }

    fn allocation(bytes: usize) -> Buffer {
        crate::test::allocations(0..bytes.next_multiple_of(512), bytes.next_multiple_of(512))
            .alloc(Kind::Dram, bytes)
            .unwrap()
    }

    /// A write excludes any overlapping access on a shared chip, however earlier ones ended, and
    /// coalescing views never widens or narrows what is held.
    #[test]
    fn excludes_overlapping_writes() {
        let buffer = allocation(256);
        let first = buffer.slice(..128).unwrap();
        for (bytes, chips, access, conflict) in [
            (0..128, 1, Read, true),
            (0..128, 1, Write, true),
            (127..256, 3, Read, true),
            (128..256, 1, Write, false),
            (0..128, 2, Write, false),
            (128..128, 1, Write, false),
        ] {
            let mut held = Lease::new(1);
            let mut next = Lease::new(1);
            held.extend([(first, Write)]).unwrap();
            assert_eq!(
                next.extend([(view(&buffer, bytes, chips), access)]),
                if conflict { Err(Error::Busy) } else { Ok(()) },
            );
        }
        let mut readers = Lease::new(2);
        readers
            .extend([(view(&buffer, 0..256, 1), Read), (view(&buffer, 0..256, 3), Read)])
            .unwrap();
        readers.clear();
        let [left, middle, right] = [0..64, 64..128, 128..192].map(|bytes| buffer.slice(bytes).unwrap());
        let (mut outer, mut reader, mut next) = (Lease::new(2), Lease::new(1), Lease::new(1));
        outer.extend([(left, Write), (right, Write)]).unwrap();
        reader.extend([(middle, Read)]).unwrap();
        outer.clear();
        next.extend([(middle, Read)]).unwrap();
        outer.extend([(right, Write), (left, Write)]).unwrap();
        next.clear();
        assert_eq!(next.extend([(middle, Write)]), Err(Error::Busy));
        reader.clear();
        next.extend([(middle, Write)]).unwrap();
        outer.clear();
        assert_eq!(reader.extend([(middle, Read)]), Err(Error::Busy));
        next.clear();
        reader.extend([(buffer.slice(..).unwrap(), Write)]).unwrap();

        // Contiguous views of one allocation hold one record that guards their whole span.
        {
            let buffer = allocation(256);
            let [left, right] = [0..64, 64..128].map(|bytes| buffer.slice(bytes).unwrap());
            let mut held = Lease::new(1);
            held.extend([(left, Write), (right, Write)]).unwrap();
            let mut next = Lease::new(1);
            assert_eq!(next.extend([(view(&buffer, 100..110, 1), Read)]), Err(Error::Busy));
            next.extend([(view(&buffer, 128..256, 1), Write)]).unwrap();
        }

        // Adjacent independent allocations keep separate ownership and access lists.
        {
            let space = crate::test::allocations(0..4096, 4096);
            let mut first = space.alloc(Kind::Dram, 512).unwrap();
            let mut second = space.alloc(Kind::Dram, 512).unwrap();
            let (first, second) = if first.addr() < second.addr() {
                (&mut first, &mut second)
            } else {
                (&mut second, &mut first)
            };
            assert_eq!(first.addr() + first.len(), second.addr());
            assert!(!first.join(second));
            let _first = Lease::acquire([(first.slice(..).unwrap(), Write)]).unwrap();
            assert!(Lease::acquire([(second.slice(..).unwrap(), Write)]).is_ok());
        }

        // A launch holding joined blocks: a transfer into a covered block is Busy, one into the
        // gap between blocks is admitted, and the launch's own alias of the blocks is Busy.
        let coalesce = |mut buffers: Vec<Buffer>| {
            buffers.dedup_by(|next, previous| previous.join(next));
            buffers
        };
        let buffer = allocation(4096);
        for (blocks, transfer, admitted) in [
            ([0..512, 512..1024], 700..800, false),
            ([0..512, 1024..1536], 512..1024, true),
        ] {
            let behind = coalesce(blocks.map(|block| buffer.alias(block).unwrap()).into());
            let _launch = Lease::acquire(behind.iter().map(|buffer| (buffer.slice(..).unwrap(), Write))).unwrap();
            let transfer = Lease::acquire([(buffer.slice(transfer).unwrap(), Write)]);
            assert_eq!(transfer.err(), (!admitted).then_some(Error::Busy));
        }
        let behind = coalesce(vec![
            buffer.alias(..512).unwrap(),
            buffer.alias(512..1024).unwrap(),
            buffer.alias(256..768).unwrap(),
        ]);
        let launch = Lease::acquire(behind.iter().map(|buffer| (buffer.slice(..).unwrap(), Write)));
        assert!(matches!(launch, Err(Error::Busy)));
    }

    /// A refused batch holds none of its regions and keeps those acquired before it.
    #[test]
    fn rolls_back_rejected_batches() {
        let buffers = [allocation(256), allocation(256)];
        let [first, second] = buffers.each_ref().map(|buffer| buffer.slice(..).unwrap());
        let mut held = Lease::new(1);
        held.extend([(second, Write)]).unwrap();
        let mut next = Lease::new(2);
        next.extend([(first, Write)]).unwrap();
        assert_eq!(next.extend([(second, Read)]), Err(Error::Busy));
        next.extend(std::iter::empty()).unwrap();
        let mut probe = Lease::new(1);
        assert_eq!(probe.extend([(first, Read)]), Err(Error::Busy));
        next.clear();
        held.clear();
        assert_eq!(probe.extend([(first, Read), (second, Read)]), Err(Error::Capacity),);
        next.extend([(first, Write)]).unwrap();
        next.extend([(second, Write)]).unwrap();
        next.clear();
        assert_eq!(next.extend([(first, Read), (first, Write)]), Err(Error::Busy),);
        probe.extend([(first, Write)]).unwrap();
    }

    /// Once work fails, the memory it held refuses every access and wait with `Poisoned`, and its
    /// device refuses later launches and transfers; other allocations stay usable.
    #[tokio::test]
    async fn poisons_memory_held_by_failed_work() {
        let buffer = allocation(256);
        let other = allocation(256);
        let mut held = Lease::new(1);
        held.extend([(buffer.slice(..128).unwrap(), Write)]).unwrap();
        held.poison();
        held.clear();
        let mut next = Lease::new(1);
        assert_eq!(next.extend([(view(&buffer, 128..256, 2), Read)]), Err(Error::Poisoned));
        next.extend([(other.slice(..).unwrap(), Write)]).unwrap();

        // Poisoning wakes the waits on its allocation, which then fail, as later ones do at once.
        {
            let buffer = allocation(256);
            let wakes = Arc::new(Wakes::default());
            let mut held = Lease::new(1);
            held.extend([(buffer.slice(..).unwrap(), Write)]).unwrap();
            let mut read = buffer.ready(Read);
            assert!(poll(&mut read, &wakes).is_pending());
            held.poison();
            assert_eq!(woken(&wakes), 1);
            assert_eq!(poll(&mut read, &wakes), Poll::Ready(Err(Error::Poisoned)));
            held.clear();
            assert_eq!(poll(&mut buffer.ready(Read), &wakes), Poll::Ready(Err(Error::Poisoned)));
        }

        // A hardware failure poisons an ordered launch's device and host operands, and the device
        // refuses later launches with that failure.
        {
            use crate::Argument;
            use crate::device::Contention;
            let (device, _) = crate::test::running(&[0], 8);
            // SAFETY: the stub has no memory accesses on any cluster.
            let function = unsafe { crate::test::function(&device).ordered() };
            let buffer = device.alloc(Kind::Dram, 256).unwrap();
            let mut host = crate::test::mapped(&device, vec![0], 4096);
            drop(
                function
                    .launch([
                        Argument::Write(buffer.slice(..).unwrap()),
                        Argument::Host(host.slice_mut(..)),
                    ])
                    .unwrap(),
            );
            let error = Error::Device("lost completion".into());
            device.driver.poison.set(error.clone());
            device.launches.complete(Contention::Skip);
            assert_eq!(function.launch([]).err(), Some(error));
            assert_eq!(
                crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Read)]).err(),
                Some(Error::Poisoned)
            );
            assert_eq!(
                host.slice(..).backing.unwrap().lease(Access::Read).err(),
                Some(Error::Poisoned)
            );
            device.driver.stop();
        }

        // A failed transfer answers its observer with the failure, and its queue refuses reuse.
        {
            use crate::driver::dma::fixture::{finish, push, tag};
            use crate::driver::dma::{DmaQueue, Transport, fake::Fake};
            use std::task::Context;
            let buffer = allocation(1);
            let host = crate::test::pinned();
            let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
            let mut transfer = queue.acquire().await.unwrap();
            push(&mut transfer, &buffer, Some(&host));
            let tag = tag(&transfer);
            let mut observer = std::pin::pin!(transfer.submit());
            let mut context = Context::from_waker(Waker::noop());
            assert_eq!(observer.as_mut().poll(&mut context), Poll::Pending);
            assert!(
                finish(&queue, tag, Err(Error::Poisoned)).is_some(),
                "completion wakes the observer"
            );
            assert!(queue.acquire().await.is_err());
            assert!(finish(&queue, tag, Ok(())).is_none());
            assert!(queue.poison().get().is_some());
            assert_eq!(observer.as_mut().poll(&mut context), Poll::Ready(Err(Error::Poisoned)));
        }
    }
}
