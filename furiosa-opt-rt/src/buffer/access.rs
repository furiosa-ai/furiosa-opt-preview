//! Allocation access retained until physical completion.
//!
//! Records are preallocated per launch or transfer slot and linked into their allocation's list in
//! place, so admitting work allocates nothing. Each record carries its admission number, so a wait
//! for the work admitted before it can skip later admissions.

// allow: SIZE_OK — Intrusive list mutation and inline alias tests share allocation-lock invariants.

use std::cell::{Cell, UnsafeCell};
use std::ops::Range;
use std::ptr;
use std::sync::{Arc, PoisonError};
use std::task::{Poll, Waker};

use super::{Access, Allocation, BufferSlice};
use crate::memory::Kind;
use crate::{Error, Result};

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
        if allocation.kind == Kind::DramShared && allocation.allocations.topology.chips > 1 {
            Self::Unordered
        } else {
            self
        }
    }
}

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
            && bytes.start < self.bytes.end
            && self.bytes.start < bytes.end
    }
}

/// Unused records are unlinked: both neighbors are null.
struct Record {
    region: UnsafeCell<Option<Region>>,
    prev: Cell<*const Record>,
    next: Cell<*const Record>,
}

// SAFETY: records are published only in a boxed slice; moving Holds never moves a linked record.
unsafe impl Send for Record {}
// SAFETY: list access uses the allocation mutex. Only its exclusive Holds owner changes region;
// readers of that region finish under the same mutex before it is unlinked and cleared.
unsafe impl Sync for Record {}

pub(crate) struct Holds {
    records: Box<[Record]>,
    len: usize,
    pub(crate) owner: Owner,
}

impl Holds {
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

    /// Adds regions to this batch; a conflict releases only the regions this call added, since
    /// the earlier ones guard work already queued.
    pub(crate) fn acquire<'a>(&mut self, regions: impl IntoIterator<Item = (BufferSlice<'a>, Access)>) -> Result<()> {
        match self.owner {
            Owner::Ordered(_) => assert_eq!(self.len, 0, "an ordered owner admits one complete batch"),
            Owner::Unordered => {}
        }
        let len = self.len;
        let result = (|| {
            let mut regions = regions.into_iter().peekable();
            while let Some((view, access)) = regions.next() {
                let buffer = view.buffer;
                let owner = self.owner.on(&buffer.allocation);
                let chips = view.chips;
                let start = buffer.offset + view.offset;
                let mut bytes = start..start + view.len;
                // Contiguous views of one allocation, as a slice of blocks binds, share one record.
                while let Some((next, _)) = regions.next_if(|(next, next_access)| {
                    Arc::ptr_eq(&next.buffer.allocation, &buffer.allocation)
                        && next.chips == chips
                        && *next_access == access
                        && next.buffer.offset + next.offset == bytes.end
                }) {
                    bytes.end += next.len;
                }
                // Nothing panics while a list is half updated, so a poisoned lock still guards a
                // whole list.
                let mut active = buffer.allocation.active.lock().unwrap_or_else(PoisonError::into_inner);
                if active.poisoned {
                    return Err(Error::Poisoned);
                }
                if bytes.is_empty() || chips == 0 {
                    continue;
                }
                let record = self.records.get(self.len).ok_or(Error::Capacity)?;
                let overlaps = match owner {
                    Owner::Ordered(_) => {
                        active.unordered != 0
                            && bytes.start < active.unordered_bytes.end
                            && active.unordered_bytes.start < bytes.end
                    }
                    Owner::Unordered => bytes.start < active.bytes.end && active.bytes.start < bytes.end,
                };
                if (access == Access::Write || active.writers != 0)
                    && overlaps
                    && active.regions().any(|region| {
                        let exempt = match (owner, region.owner.get()) {
                            (Owner::Ordered(left), Owner::Ordered(right)) => left != right,
                            (Owner::Unordered, _) | (_, Owner::Unordered) => false,
                        };
                        !exempt && region.conflicts(&bytes, chips, access)
                    })
                {
                    return Err(Error::Busy);
                }
                if active.head.is_null() {
                    active.bytes = bytes.clone();
                } else {
                    active.bytes.start = active.bytes.start.min(bytes.start);
                    active.bytes.end = active.bytes.end.max(bytes.end);
                }
                active.writers += usize::from(access == Access::Write);
                if active.unordered == 0 {
                    active.unordered_bytes = bytes.clone();
                } else {
                    active.unordered_bytes.start = active.unordered_bytes.start.min(bytes.start);
                    active.unordered_bytes.end = active.unordered_bytes.end.max(bytes.end);
                }
                active.unordered += 1;
                // SAFETY: this unused record is exclusively owned; publication uses its allocation lock.
                unsafe {
                    *record.region.get() = Some(Region {
                        allocation: Arc::clone(&buffer.allocation),
                        bytes,
                        chips,
                        access,
                        // Batch members stay non-exempt until intra-launch aliases have been checked.
                        owner: Cell::new(Owner::Unordered),
                        sequence: active.admitted,
                    });
                    if let Some(head) = active.head.as_ref() {
                        head.prev.set(ptr::from_ref(record));
                    }
                }
                record.next.set(active.head);
                active.head = ptr::from_ref(record);
                active.admitted += 1;
                self.len += 1;
            }
            Ok(())
        })();
        result.inspect_err(|_| self.truncate(len))?;
        match self.owner {
            Owner::Ordered(_) => {
                for record in &self.records[..self.len] {
                    // SAFETY: this Holds owns the region; its allocation mutex excludes list readers.
                    let region = unsafe { &*record.region.get() }.as_ref().expect("owned access");
                    let mut active = region.allocation.active.lock().unwrap_or_else(PoisonError::into_inner);
                    let owner = self.owner.on(&region.allocation);
                    region.owner.set(owner);
                    match owner {
                        Owner::Ordered(_) => active.unordered -= 1,
                        Owner::Unordered => {}
                    }
                }
            }
            Owner::Unordered => {}
        }
        Ok(())
    }

    pub(crate) fn capacity(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn clear(&mut self) {
        self.truncate(0);
    }

    fn truncate(&mut self, len: usize) {
        for record in self.records[len..self.len].iter().rev() {
            // SAFETY: Holds owns this region until unlink; no shared Holds method can run concurrently.
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

    /// A failed operation can leave its allocation unsafe even after its records are released.
    pub(crate) fn poison(&self) {
        for record in &self.records[..self.len] {
            // SAFETY: shared Holds methods preserve region; other owners may change only its links.
            let region = unsafe { &*record.region.get() }.as_ref().expect("owned access");
            let mut active = region.allocation.active.lock().unwrap_or_else(PoisonError::into_inner);
            active.poisoned = true;
            let waiters = active.waiters.take();
            drop(active);
            waiters.wake();
        }
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

impl Drop for Holds {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use Access::{Read, Write};

    use super::*;
    use crate::buffer::Buffer;
    use crate::memory::Kind;

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

    /// A write excludes any overlapping access on a shared chip, however earlier ones ended.
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
            let mut held = Holds::new(1);
            let mut next = Holds::new(1);
            held.acquire([(first, Write)]).unwrap();
            assert_eq!(
                next.acquire([(view(&buffer, bytes, chips), access)]),
                if conflict { Err(Error::Busy) } else { Ok(()) },
            );
        }
        let mut readers = Holds::new(2);
        readers
            .acquire([(view(&buffer, 0..256, 1), Read), (view(&buffer, 0..256, 3), Read)])
            .unwrap();
        readers.clear();
        let [left, middle, right] = [0..64, 64..128, 128..192].map(|bytes| buffer.slice(bytes).unwrap());
        let (mut outer, mut reader, mut next) = (Holds::new(2), Holds::new(1), Holds::new(1));
        outer.acquire([(left, Write), (right, Write)]).unwrap();
        reader.acquire([(middle, Read)]).unwrap();
        outer.clear();
        next.acquire([(middle, Read)]).unwrap();
        outer.acquire([(right, Write), (left, Write)]).unwrap();
        next.clear();
        assert_eq!(next.acquire([(middle, Write)]), Err(Error::Busy));
        reader.clear();
        next.acquire([(middle, Write)]).unwrap();
        outer.clear();
        assert_eq!(reader.acquire([(middle, Read)]), Err(Error::Busy));
        next.clear();
        reader.acquire([(buffer.slice(..).unwrap(), Write)]).unwrap();
    }

    /// A refused batch holds none of its regions and keeps those acquired before it.
    #[test]
    fn rolls_back_rejected_batches() {
        let buffers = [allocation(256), allocation(256)];
        let [first, second] = buffers.each_ref().map(|buffer| buffer.slice(..).unwrap());
        let mut held = Holds::new(1);
        held.acquire([(second, Write)]).unwrap();
        let mut next = Holds::new(2);
        next.acquire([(first, Write)]).unwrap();
        assert_eq!(next.acquire([(second, Read)]), Err(Error::Busy));
        next.acquire(std::iter::empty()).unwrap();
        let mut probe = Holds::new(1);
        assert_eq!(probe.acquire([(first, Read)]), Err(Error::Busy));
        next.clear();
        held.clear();
        assert_eq!(probe.acquire([(first, Read), (second, Read)]), Err(Error::Capacity),);
        next.acquire([(first, Write)]).unwrap();
        next.acquire([(second, Write)]).unwrap();
        next.clear();
        assert_eq!(next.acquire([(first, Read), (first, Write)]), Err(Error::Busy),);
        probe.acquire([(first, Write)]).unwrap();
    }

    /// Contiguous views of one allocation hold one record that guards their whole span.
    #[test]
    fn merges_contiguous_views() {
        let buffer = allocation(256);
        let [left, right] = [0..64, 64..128].map(|bytes| buffer.slice(bytes).unwrap());
        let mut held = Holds::new(1);
        held.acquire([(left, Write), (right, Write)]).unwrap();
        let mut next = Holds::new(1);
        assert_eq!(next.acquire([(view(&buffer, 100..110, 1), Read)]), Err(Error::Busy));
        next.acquire([(view(&buffer, 128..256, 1), Write)]).unwrap();
    }

    /// After a failed operation, its allocation refuses every access with `Poisoned`.
    #[test]
    fn poisons_the_allocation() {
        let buffer = allocation(256);
        let other = allocation(256);
        let mut held = Holds::new(1);
        held.acquire([(buffer.slice(..128).unwrap(), Write)]).unwrap();
        held.poison();
        held.clear();
        let mut next = Holds::new(1);
        assert_eq!(next.acquire([(view(&buffer, 128..256, 2), Read)]), Err(Error::Poisoned));
        next.acquire([(other.slice(..).unwrap(), Write)]).unwrap();
    }

    /// Non-ordered accesses exclude ordered launches even amid many exempt records.
    #[test]
    fn ordered_exclusion() {
        let buffer = allocation(256);
        let whole = buffer.slice(..).unwrap();
        let mut first = Holds::new(1);
        first.owner = Owner::Ordered(1);
        first.acquire([(whole, Write)]).unwrap();
        let mut transfer = Holds::new(1);
        assert_eq!(transfer.acquire([(whole, Read)]), Err(Error::Busy));
        first.clear();
        transfer.acquire([(buffer.slice(128..).unwrap(), Write)]).unwrap();
        first.acquire([(buffer.slice(..128).unwrap(), Write)]).unwrap();
        let mut second = Holds::new(1);
        second.owner = Owner::Ordered(2);
        assert_eq!(second.acquire([(whole, Write)]), Err(Error::Busy));
        transfer.clear();
        second.acquire([(whole, Write)]).unwrap();
        assert_eq!(transfer.acquire([(whole, Read)]), Err(Error::Busy));
        first.poison();
        first.clear();
        second.clear();
        assert_eq!(second.acquire([(whole, Write)]), Err(Error::Poisoned));
    }

    /// Counts its wakes.
    #[derive(Default)]
    struct Wakes(std::sync::atomic::AtomicUsize);

    impl std::task::Wake for Wakes {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn poll(wait: &mut (impl Future<Output = Result<()>> + Unpin), wakes: &Arc<Wakes>) -> Poll<Result<()>> {
        let waker = Waker::from(Arc::clone(wakes));
        std::pin::Pin::new(wait).poll(&mut std::task::Context::from_waker(&waker))
    }

    fn woken(wakes: &Wakes) -> usize {
        wakes.0.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A read waits for overlapping writers on a shared chip, not for readers; a write waits for
    /// both, and each wakes when the last record it waits for leaves.
    #[test]
    fn waits_for_conflicting_records() {
        let buffer = allocation(256);
        let wakes = Arc::new(Wakes::default());
        let (mut reader, mut writer) = (Holds::new(1), Holds::new(1));
        reader.acquire([(buffer.slice(..128).unwrap(), Read)]).unwrap();
        assert_eq!(
            poll(&mut buffer.slice(..).unwrap().ready(Read), &wakes),
            Poll::Ready(Ok(()))
        );
        writer.acquire([(buffer.slice(128..).unwrap(), Write)]).unwrap();
        for (bytes, chips, access, waits) in [
            (0..128, 1, Read, false),
            (128..256, 2, Read, false),
            (100..129, 1, Read, true),
            (0..64, 1, Write, true),
            (128..256, 1, Write, true),
        ] {
            let ready = poll(&mut view(&buffer, bytes, chips).ready(access), &wakes);
            assert_eq!(ready.is_pending(), waits, "{access:?}");
        }
        let mut write = buffer.slice(..).unwrap().ready(Write);
        assert!(poll(&mut write, &wakes).is_pending());
        writer.clear();
        assert!(woken(&wakes) > 0);
        assert!(poll(&mut write, &wakes).is_pending());
        reader.clear();
        assert_eq!(poll(&mut write, &wakes), Poll::Ready(Ok(())));
    }

    /// A wait covers the records admitted before it, not those admitted while it waits.
    #[test]
    fn waits_for_earlier_records_only() {
        let buffer = allocation(256);
        let wakes = Arc::new(Wakes::default());
        let (mut first, mut later) = (Holds::new(1), Holds::new(1));
        first.acquire([(buffer.slice(..).unwrap(), Read)]).unwrap();
        let mut write = buffer.slice(..).unwrap().ready(Write);
        later.acquire([(buffer.slice(..).unwrap(), Read)]).unwrap();
        assert!(poll(&mut write, &wakes).is_pending());
        first.clear();
        assert_eq!(poll(&mut write, &wakes), Poll::Ready(Ok(())));
        assert!(poll(&mut buffer.ready(Write), &wakes).is_pending());
    }

    /// Poisoning wakes the waits on its allocation, which then fail, as later ones do at once.
    #[test]
    fn poisoned_waits_fail() {
        let buffer = allocation(256);
        let wakes = Arc::new(Wakes::default());
        let mut held = Holds::new(1);
        held.acquire([(buffer.slice(..).unwrap(), Write)]).unwrap();
        let mut read = buffer.ready(Read);
        assert!(poll(&mut read, &wakes).is_pending());
        held.poison();
        assert_eq!(woken(&wakes), 1);
        assert_eq!(poll(&mut read, &wakes), Poll::Ready(Err(Error::Poisoned)));
        held.clear();
        assert_eq!(poll(&mut buffer.ready(Read), &wakes), Poll::Ready(Err(Error::Poisoned)));
    }

    /// Reports CPU admission cost for a pool-sized batch, excluding allocation and retirement.
    #[test]
    #[ignore = "CPU admission benchmark"]
    fn ordered_admission_bench() {
        const COUNT: usize = 4096;
        let buffer = allocation(COUNT);
        let mut first = Holds::new(COUNT);
        let mut second = Holds::new(COUNT);
        first.owner = Owner::Ordered(1);
        second.owner = Owner::Ordered(2);
        let regions = || (0..COUNT).map(|index| (buffer.slice(index..index + 1).unwrap(), Write));
        first.acquire(regions()).unwrap();
        let mut samples = [0; 21];
        for sample in &mut samples {
            let started = std::time::Instant::now();
            second.acquire(regions()).unwrap();
            *sample = started.elapsed().as_nanos();
            second.clear();
        }
        samples.sort_unstable();
        std::println!(
            "4096 held + 4096 admitted: median={} ns, min={} ns, max={} ns",
            samples[10],
            samples[0],
            samples[20]
        );
    }
}
