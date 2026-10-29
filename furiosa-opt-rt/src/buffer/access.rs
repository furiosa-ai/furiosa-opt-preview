//! Which admitted work holds each allocation's bytes, until the hardware is done with them.
//!
//! A lease holds host or device regions through completion; admission order bounds conflicting waits.

use std::cell::{Cell, UnsafeCell};
use std::collections::BTreeMap;
use std::ops::Range;
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Poll, Waker};

use super::{Access, Allocation, BufferSlice};
use crate::memory::Kind;
use crate::pinned::{DmaBuf, Registration};
use crate::{Error, Result};

/// Accesses held until dropped: reads exclude overlapping writes, writes every overlapping access.
pub struct Lease {
    records: Box<[Record]>,
    len: usize,
    owner: Owner,
}

impl Lease {
    /// Holds `holds`, each with its access; [`Error::Busy`] when two of them or another lease
    /// conflict, and then this one holds nothing.
    pub fn acquire<'a, H: Into<Hold<'a>>>(holds: impl IntoIterator<Item = H>) -> Result<Self> {
        let holds: Vec<Hold<'a>> = holds.into_iter().map(Into::into).collect();
        let mut lease = Self::new(holds.len());
        lease.extend(holds)?;
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
    pub(crate) fn extend<T: Into<Region>>(&mut self, regions: impl IntoIterator<Item = T>) -> Result<()> {
        if let Owner::Ordered(_) = self.owner {
            assert_eq!(self.len, 0, "an ordered owner admits one complete batch");
        }
        let mark = self.len;
        coalesce(regions.into_iter().map(Into::into))
            .try_for_each(|request| self.hold(request, Queued::Excludes).map(drop))
            .inspect_err(|_| self.truncate(mark))?;
        self.seal();
        Ok(())
    }

    /// Adds a launch's regions as [`Self::extend`] does, except that queued work on their bytes does
    /// not refuse them: they wait for it, and work queued after them waits for them in turn.
    pub(crate) fn admit<T: Into<Region>>(&mut self, regions: impl IntoIterator<Item = T>) -> Result<()> {
        if let Owner::Ordered(_) = self.owner {
            assert_eq!(self.len, 0, "an ordered owner admits one complete batch");
        }
        let mark = self.len;
        // Batches queue under the same lock, so in every allocation each one is all before these
        // regions or all after them, which leaves no cycle of work waiting for each other.
        let admission = Admission::begin();
        let behind = coalesce(regions.into_iter().map(Into::into))
            .try_fold(false, |behind, request| {
                self.hold(request, Queued::Precedes).map(|waits| waits | behind)
            })
            .inspect_err(|_| self.truncate(mark))?;
        self.seal();
        drop(admission);
        if behind {
            self.follow(mark).inspect_err(|_| self.truncate(mark))?;
        }
        Ok(())
    }

    /// A failed operation can leave its allocation unsafe even after its records are released.
    pub(crate) fn poison(&self) {
        (self.regions())
            .filter(|region| region.access == Access::Write)
            .for_each(Region::poison);
    }

    pub(crate) fn clear(&mut self) {
        self.truncate(0);
    }

    pub(crate) fn capacity(&self) -> usize {
        self.records.len()
    }

    /// Adds `regions` of queued work without refusing conflicts: [`Self::ready`] waits for them.
    /// `admission` keeps every region of one batch ahead of every later batch's, in every
    /// allocation, so two batches never wait for each other.
    pub(crate) fn enqueue(&mut self, admission: &Admission, regions: impl IntoIterator<Item = Region>) -> Result<()> {
        let mark = self.len;
        let mut regions = coalesce(regions).collect::<Vec<_>>();
        // Grouped by allocation, the lease's records there take one lock, one sequence and one list.
        regions.sort_unstable_by_key(|region| ptr::from_ref(region.allocation.active()).addr());
        (regions.chunk_by(|left, right| left.allocation.same(&right.allocation)))
            .try_for_each(|group| {
                let mut active = group[0]
                    .allocation
                    .active()
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                if active.poisoned {
                    return Err(Error::Poisoned);
                }
                let requests = group
                    .iter()
                    .filter(|request| !request.bytes.is_empty() && request.chips != 0);
                if requests.clone().count() > self.capacity() - self.len {
                    return Err(Error::Capacity);
                }
                let sequence = active.admitted;
                for request in requests {
                    let region = Region {
                        owner: Cell::new(Owner::Unordered),
                        sequence,
                        batch: Some(*admission.batch),
                        ..request.clone()
                    };
                    active.insert(&self.records[self.len], region);
                    self.len += 1;
                }
                Ok(())
            })
            .inspect_err(|_| self.truncate(mark))
    }

    /// Resolves once the conflicting work admitted before this queued lease released its regions.
    /// Its batch's other leases do not count: a transfer checks each slot after the earlier ones end.
    pub(crate) fn ready(&self) -> impl Future<Output = Result<()>> + Send + '_ {
        std::future::poll_fn(|context| {
            let mut regions = self.regions().collect::<Vec<_>>();
            regions.sort_unstable_by_key(|region| ptr::from_ref(region.allocation.active()).addr());
            for group in regions.chunk_by(|left, right| left.allocation.same(&right.allocation)) {
                let mut active = group[0]
                    .allocation
                    .active()
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                if active.poisoned {
                    return Poll::Ready(Err(Error::Poisoned));
                }
                if active.blocks(group, || {}) {
                    active.waiters.register(context.waker());
                    return Poll::Pending;
                }
            }
            Poll::Ready(Ok(()))
        })
    }

    /// The regions this batch holds.
    fn regions(&self) -> impl Iterator<Item = &Region> {
        self.records[..self.len].iter().map(|record| {
            // SAFETY: shared Lease methods preserve region; other owners may change only its links.
            unsafe { &*record.region.get() }.as_ref().expect("owned access")
        })
    }

    /// Holds `request` in this batch's next record, unless work already holding its bytes refuses
    /// it; true when queued work holds some of them, which `queued` says it then waits for.
    fn hold(&mut self, request: Region, queued: Queued) -> Result<bool> {
        // Nothing panics while a list is half updated, so a poisoned lock still guards a whole list.
        let mut active = request
            .allocation
            .active()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if active.poisoned {
            return Err(Error::Poisoned);
        }
        if request.bytes.is_empty() || request.chips == 0 {
            return Ok(false);
        }
        let record = self.records.get(self.len).ok_or(Error::Capacity)?;
        if active.refuses(&request, self.owner.on(&request.allocation)) {
            return Err(Error::Busy);
        }
        let behind = active.blocker(&request, active.admitted).is_some();
        if behind && queued == Queued::Excludes {
            return Err(Error::Busy);
        }
        let sequence = active.admitted;
        active.insert(
            record,
            Region {
                allocation: request.allocation.clone(),
                bytes: request.bytes,
                chips: request.chips,
                access: request.access,
                blocks: request.blocks,
                // Batch members stay non-exempt until intra-launch aliases have been checked.
                owner: Cell::new(Owner::Unordered),
                sequence,
                batch: None,
            },
        );
        self.len += 1;
        Ok(behind)
    }

    /// Blocks until the queued work admitted before this batch's records from `mark` on released
    /// the bytes they conflict with; [`Error::Poisoned`] once that work left them unknown.
    fn follow(&self, mark: usize) -> Result<()> {
        for region in self.regions().skip(mark) {
            let mut active = region
                .allocation
                .active()
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let idle = Arc::clone(&active.idle);
            // Earlier records only leave, so each blocker is looked for once, not at every wake.
            while let Some(blocker) = active.blocker(region, region.sequence) {
                while !active.poisoned && active.batches.contains_key(&blocker) {
                    active = idle.wait(active).unwrap_or_else(PoisonError::into_inner);
                }
                if active.poisoned {
                    return Err(Error::Poisoned);
                }
            }
        }
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
            let mut active = region
                .allocation
                .active()
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
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
                let mut active = region
                    .allocation
                    .active()
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let (prev, next) = (record.prev.replace(ptr::null()), record.next.replace(ptr::null()));
                // SAFETY: every neighbor remains linked and alive under this allocation's mutex.
                match (unsafe { prev.as_ref() }, region.batch) {
                    (Some(prev), _) => prev.next.set(next),
                    (None, None) => active.head = next,
                    (None, Some(batch)) if next.is_null() => {
                        active.batches.remove(&(batch, region.sequence));
                    }
                    (None, Some(batch)) => {
                        active.batches.insert((batch, region.sequence), next);
                    }
                }
                if let Some(next) = unsafe { next.as_ref() } {
                    next.prev.set(prev);
                }
                active.writers -= usize::from(region.access == Access::Write);
                match region.owner.get() {
                    Owner::Unordered => active.unordered -= 1,
                    Owner::Ordered(_) => {}
                }
                if active.head.is_null() && active.batches.is_empty() {
                    active.bytes = 0..0;
                }
                active.idle.notify_all();
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
        let request = Region::from((self, access));
        let before = request
            .allocation
            .active()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .admitted;
        std::future::poll_fn(move |context| {
            let mut active = request
                .allocation
                .active()
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if active.poisoned {
                return Poll::Ready(Err(Error::Poisoned));
            }
            let blocked = (access == Access::Write || active.writers != 0)
                && overlap(&request.bytes, &active.bytes)
                && active
                    .regions()
                    .any(|region| region.sequence < before && region.conflicts(&request));
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

/// One batch of queued work being admitted. Records are numbered per allocation in the order they
/// are inserted, and a queued record waits for the lower-numbered ones of other work it conflicts
/// with; batches admitted one at a time are therefore ordered alike in every allocation, which
/// leaves no cycle of batches waiting for each other.
pub(crate) struct Admission {
    /// Holds off other batches; the count it guards numbers this one.
    batch: MutexGuard<'static, u64>,
}

impl Admission {
    /// Waits until no other batch is being admitted.
    pub(crate) fn begin() -> Self {
        static ADMITTED: Mutex<u64> = Mutex::new(0);
        let mut batch = ADMITTED.lock().unwrap_or_else(PoisonError::into_inner);
        *batch += 1;
        Self { batch }
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

/// What queued work on bytes a hold conflicts with means for it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Queued {
    /// It refuses the hold, as work admitted at once does.
    Excludes,
    /// It runs first: the hold is a launch's, which waits for it.
    Precedes,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Owner {
    Ordered(u64),
    #[default]
    Unordered,
}

impl Owner {
    fn on(self, allocation: &Memory) -> Self {
        match allocation {
            Memory::Device(allocation)
                if allocation.kind == Kind::DramShared && allocation.allocator.topology().chips > 1 =>
            {
                Self::Unordered
            }
            Memory::Device(_) | Memory::Host(_, _) => self,
        }
    }
}

#[derive(Default)]
pub(crate) struct Active {
    /// The records of work admitted at once, newest first.
    head: *const Record,
    /// The records of each queued lease, newest first, by its batch and sequence.
    batches: BTreeMap<(u64, u64), *const Record>,
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
    idle: Arc<Condvar>,
}

// SAFETY: the allocation mutex serializes every list access; linked records remain alive and fixed.
unsafe impl Send for Active {}

impl Active {
    /// Whether work admitted at once refuses `owner`'s `request` by conflicting with it, unless both
    /// are distinct ordered owners of memory they order alike.
    fn refuses(&self, request: &Region, owner: Owner) -> bool {
        let overlaps = match owner {
            Owner::Ordered(_) => match request.allocation {
                Memory::Host(_, _) => overlap(&request.bytes, &self.bytes),
                Memory::Device(_) => self.unordered != 0 && overlap(&request.bytes, &self.unordered_bytes),
            },
            Owner::Unordered => overlap(&request.bytes, &self.bytes),
        };
        (request.access == Access::Write || self.writers != 0)
            && overlaps
            && self.list(self.head).any(|region| {
                let exempt = match (owner, region.owner.get()) {
                    (Owner::Ordered(left), Owner::Ordered(right)) => {
                        left != right && request.allocation.ordered(&region.allocation)
                    }
                    (Owner::Unordered, _) | (_, Owner::Unordered) => false,
                };
                !exempt && region.conflicts(request)
            })
    }

    /// A queued lease admitted before `sequence`, by its batch and sequence, that holds bytes
    /// `request` conflicts with, if one does.
    fn blocker(&self, request: &Region, sequence: u64) -> Option<(u64, u64)> {
        if self.batches.is_empty()
            || (request.access == Access::Read && self.writers == 0)
            || !overlap(&request.bytes, &self.bytes)
        {
            return None;
        }
        (self.batches.iter())
            .find(|&(&(_, queued), &head)| queued < sequence && self.list(head).any(|other| other.conflicts(request)))
            .map(|(&key, _)| key)
    }

    /// Links `record`, now holding `region`, at the head of its list and widens the envelopes over
    /// it.
    fn insert(&mut self, record: &Record, region: Region) {
        self.bytes = if self.head.is_null() && self.batches.is_empty() {
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
        let head = match region.batch {
            Some(batch) => self.batches.entry((batch, region.sequence)).or_insert(ptr::null()),
            None => &mut self.head,
        };
        // SAFETY: this unused record is exclusively owned; publication uses its allocation lock.
        unsafe {
            *record.region.get() = Some(region);
            if let Some(head) = (*head).as_ref() {
                head.prev.set(ptr::from_ref(record));
            }
        }
        record.next.set(*head);
        *head = ptr::from_ref(record);
        self.admitted += 1;
    }

    /// Whether `requests`, one queued lease's regions in this allocation, wait for a conflicting
    /// region of other work admitted before them; `inspect` sees each comparison of two regions.
    fn blocks(&self, requests: &[&Region], mut inspect: impl FnMut()) -> bool {
        let batch = requests[0].batch.expect("only queued work waits for readiness");
        let sequence = requests
            .iter()
            .fold(0, |sequence, request| sequence.max(request.sequence));
        // Batches are admitted in turn, so earlier ones hold the lower numbers in every allocation.
        let earlier = self.batches.range(..(batch, 0)).map(|(_, head)| head);
        (std::iter::once(&self.head).chain(earlier))
            .flat_map(|head| self.list(*head))
            .any(|other| {
                other.sequence < sequence
                    && requests.iter().any(|request| {
                        inspect();
                        other.sequence < request.sequence && other.conflicts(request)
                    })
            })
    }

    /// The linked records' regions: work admitted at once, then each queued lease's.
    fn regions(&self) -> impl Iterator<Item = &Region> {
        std::iter::once(&self.head)
            .chain(self.batches.values())
            .flat_map(|head| self.list(*head))
    }

    /// The regions of the records linked from `head`, newest first.
    fn list(&self, head: *const Record) -> impl Iterator<Item = &Region> {
        let mut next = head;
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

#[derive(Clone)]
pub(crate) struct Region {
    allocation: Memory,
    bytes: Range<usize>,
    chips: u8,
    access: Access,
    /// The blocks of `bytes` held, if not all of them.
    blocks: Option<Blocks>,
    owner: Cell<Owner>,
    /// Records of the allocation admitted before this one; a queued lease's records there share it.
    sequence: u64,
    /// The batch that queued this region; work admitted at once has none.
    batch: Option<u64>,
}

impl Region {
    /// Blocks the CPU until no admitted work holds bytes this access conflicts with; panics once
    /// the memory is poisoned, since its contents are then unknown. The CPU holds nothing after
    /// the wait: the borrow its access comes from keeps new device work off the bytes.
    pub(crate) fn wait(&self) {
        let mut active = self.allocation.active().lock().unwrap_or_else(PoisonError::into_inner);
        let idle = Arc::clone(&active.idle);
        loop {
            assert!(
                !active.poisoned,
                "DMA failed; this host allocation is no longer accessible"
            );
            if !active.refuses(self, Owner::Unordered) && active.blocker(self, u64::MAX).is_none() {
                return;
            }
            active = idle.wait(active).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Makes the allocation refuse every later access, and fails the waits on it.
    fn poison(&self) {
        let mut active = self.allocation.active().lock().unwrap_or_else(PoisonError::into_inner);
        active.poisoned = true;
        active.idle.notify_all();
        let waiters = active.waiters.take();
        drop(active);
        waiters.wake();
    }

    /// Whether this region's access and `other`'s race: either writes, on a shared chip, at a byte
    /// both hold.
    fn conflicts(&self, other: &Region) -> bool {
        let common = self.bytes.start.max(other.bytes.start)..self.bytes.end.min(other.bytes.end);
        if (self.access == Access::Read && other.access == Access::Read)
            || self.chips & other.chips == 0
            || common.is_empty()
        {
            return false;
        }
        match (&self.blocks, &other.blocks) {
            (None, None) => true,
            (Some(blocks), None) => blocks.reach(self.bytes.start, &common),
            (None, Some(blocks)) => blocks.reach(other.bytes.start, &common),
            (Some(left), Some(right)) => left.meets(self.bytes.start, right, other.bytes.start, &common),
        }
    }
}

#[derive(Clone)]
enum Memory {
    Device(Arc<Allocation>),
    Host(Arc<DmaBuf>, Option<Arc<Registration>>),
}

impl Memory {
    fn active(&self) -> &Mutex<Active> {
        match self {
            Self::Device(allocation) => &allocation.active,
            Self::Host(allocation, _) => &allocation.active,
        }
    }

    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Device(left), Self::Device(right)) => Arc::ptr_eq(left, right),
            (Self::Host(left, _), Self::Host(right, _)) => Arc::ptr_eq(left, right),
            (Self::Device(_), Self::Host(_, _)) | (Self::Host(_, _), Self::Device(_)) => false,
        }
    }

    fn ordered(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Device(_), Self::Device(_)) => true,
            (Self::Host(_, Some(left)), Self::Host(_, Some(right))) => Arc::ptr_eq(left, right),
            (Self::Host(_, None), _)
            | (_, Self::Host(_, None))
            | (Self::Device(_), Self::Host(_, Some(_)))
            | (Self::Host(_, Some(_)), Self::Device(_)) => false,
        }
    }
}

impl From<Hold<'_>> for Region {
    fn from(Hold { view, access, blocks }: Hold<'_>) -> Self {
        let start = view.buffer.offset + view.offset;
        Self {
            allocation: Memory::Device(Arc::clone(&view.buffer.allocation)),
            bytes: start..start + view.len,
            chips: view.chips,
            access,
            blocks: blocks.cloned(),
            owner: Cell::new(Owner::Unordered),
            sequence: 0,
            batch: None,
        }
    }
}

impl From<(BufferSlice<'_>, Access)> for Region {
    fn from(pair: (BufferSlice<'_>, Access)) -> Self {
        Hold::from(pair).into()
    }
}

impl Region {
    pub(crate) fn host(backing: crate::pinned::Backing<'_>, len: usize, access: Access) -> Self {
        Self {
            allocation: Memory::Host(Arc::clone(backing.dmabuf), backing.registration.cloned()),
            bytes: backing.offset..backing.offset + len,
            chips: 1,
            access,
            blocks: None,
            owner: Cell::new(Owner::Unordered),
            sequence: 0,
            batch: None,
        }
    }
}

/// Memory a lease holds: a view with its access, every byte of it or only some equal blocks of it.
#[derive(Clone, Copy, Debug)]
pub struct Hold<'a> {
    view: BufferSlice<'a>,
    access: Access,
    blocks: Option<&'a Blocks>,
}

impl<'a> Hold<'a> {
    /// Only the `blocks` of `view`, which must tile it exactly; [`Error::Invalid`] otherwise.
    pub fn blocks(view: BufferSlice<'a>, access: Access, blocks: &'a Blocks) -> Result<Self> {
        if blocks.size.checked_mul(blocks.count) != Some(view.len) {
            return Err(Error::Invalid(format!(
                "{} blocks of {} bytes do not tile a view of {} bytes",
                blocks.count, blocks.size, view.len
            )));
        }
        Ok(Self {
            view,
            access,
            blocks: Some(blocks),
        })
    }

    pub(crate) fn view(&self) -> BufferSlice<'a> {
        self.view
    }
}

/// Every byte of the view.
impl<'a> From<(BufferSlice<'a>, Access)> for Hold<'a> {
    fn from((view, access): (BufferSlice<'a>, Access)) -> Self {
        Self {
            view,
            access,
            blocks: None,
        }
    }
}

/// Some of `count` equal blocks of `size` bytes, which a [`Hold`] names in a view they tile:
/// block `i` is the view's bytes `i * size..(i + 1) * size`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blocks {
    size: usize,
    count: usize,
    /// One bit per block, block `i` at bit `i % 64` of word `i / 64`.
    members: Arc<[u64]>,
}

impl Blocks {
    /// The blocks each of `runs` covers, among `count` blocks of `size` bytes; [`Error::Invalid`]
    /// for blocks of no bytes or a run past `count`. Empty runs name nothing.
    pub fn new(size: usize, count: usize, runs: impl IntoIterator<Item = Range<usize>>) -> Result<Self> {
        if size == 0 {
            return Err(Error::Invalid("blocks must hold bytes".into()));
        }
        let mut members = vec![0u64; count.div_ceil(64)];
        for run in runs.into_iter().filter(|run| !run.is_empty()) {
            if run.end > count {
                return Err(Error::Invalid(format!("blocks {run:?} are not among {count} blocks")));
            }
            for (word, mask) in Self::words(run) {
                members[word] |= mask;
            }
        }
        Ok(Self {
            size,
            count,
            members: members.into(),
        })
    }

    /// These blocks without those of `other`; [`Error::Invalid`] unless both divide the same bytes.
    pub fn difference(&self, other: &Self) -> Result<Self> {
        if (self.size, self.count) != (other.size, other.count) {
            return Err(Error::Invalid(format!(
                "{} blocks of {} bytes and {} blocks of {} bytes",
                self.count, self.size, other.count, other.size
            )));
        }
        let members = (self.members.iter().zip(other.members.iter())).map(|(own, theirs)| own & !theirs);
        Ok(Self {
            size: self.size,
            count: self.count,
            members: members.collect(),
        })
    }

    fn contains(&self, block: usize) -> bool {
        self.members[block / 64] >> (block % 64) & 1 != 0
    }

    /// The blocks covering `bytes`, which lie inside the blocks starting at byte `start`.
    fn covering(&self, start: usize, bytes: &Range<usize>) -> Range<usize> {
        (bytes.start - start) / self.size..(bytes.end - start).div_ceil(self.size)
    }

    /// Whether a member covers a byte of `bytes`, the blocks starting at byte `start`.
    fn reach(&self, start: usize, bytes: &Range<usize>) -> bool {
        Self::words(self.covering(start, bytes)).any(|(word, mask)| self.members[word] & mask != 0)
    }

    /// Whether a member of these blocks, starting at byte `start`, and a member of `other`,
    /// starting at byte `from`, cover a byte of `bytes`, which lies inside both.
    fn meets(&self, start: usize, other: &Blocks, from: usize, bytes: &Range<usize>) -> bool {
        if start < from {
            return other.meets(from, self, start, bytes);
        }
        if self.size != other.size || !(start - from).is_multiple_of(self.size) {
            // ponytail: blocks on different grids walk one side's members, O(blocks).
            return self.held(start, bytes).any(|bytes| other.reach(from, &bytes));
        }
        // Block `i` here is block `i + shift` of `other`.
        let shift = (start - from) / self.size;
        Self::words(self.covering(start, bytes))
            .any(|(word, mask)| self.members[word] & mask & other.window(word * 64 + shift) != 0)
    }

    /// The membership of the 64 blocks from block `first` on, `first` in the lowest bit.
    fn window(&self, first: usize) -> u64 {
        let word = |index: usize| self.members.get(index).copied().unwrap_or(0);
        match (first / 64, first % 64) {
            (index, 0) => word(index),
            (index, offset) => word(index) >> offset | word(index + 1) << (64 - offset),
        }
    }

    /// The bytes of each member inside `bytes`, the blocks starting at byte `start`.
    fn held(&self, start: usize, bytes: &Range<usize>) -> impl Iterator<Item = Range<usize>> {
        let (bytes, size) = (bytes.clone(), self.size);
        (self.covering(start, &bytes))
            .filter(|&block| self.contains(block))
            .map(move |block| {
                let at = start + block * size;
                at.max(bytes.start)..(at + size).min(bytes.end)
            })
    }

    /// The membership words that hold `blocks`, each with the bits of `blocks` it holds.
    fn words(blocks: Range<usize>) -> impl Iterator<Item = (usize, u64)> {
        (blocks.start / 64..blocks.end.div_ceil(64)).map(move |word| {
            let low = blocks.start.max(word * 64) - word * 64;
            let high = blocks.end.min(word * 64 + 64) - word * 64;
            (word, (u64::MAX >> (64 - high)) & (u64::MAX << low))
        })
    }
}

fn coalesce(regions: impl IntoIterator<Item = Region>) -> impl Iterator<Item = Region> {
    let mut regions = regions.into_iter().peekable();
    std::iter::from_fn(move || {
        let mut request = regions.next()?;
        while let Some(next) = regions.next_if(|next| {
            request.blocks.is_none()
                && next.blocks.is_none()
                && next.allocation.same(&request.allocation)
                && next.chips == request.chips
                && next.access == request.access
                && next.bytes.start == request.bytes.end
        }) {
            request.bytes.end = next.bytes.end;
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
        crate::test::allocator(0..bytes.next_multiple_of(512), bytes.next_multiple_of(512))
            .alloc(Kind::Dram, bytes)
            .unwrap()
    }

    /// Readiness compares each fragment of a transfer with each earlier region of other work in its
    /// allocation and with nothing else: F fragments beside M such regions cost F·M comparisons.
    #[test]
    fn compares_fragments_with_other_work() {
        let buffer = allocation(8192);
        // Separate fragments, so none coalesce; other work holds later bytes, conflicting with none.
        let fragment = |index: usize| Region::from((buffer.slice(8 * index..8 * index + 4).unwrap(), Write));
        let other = |index: usize| (buffer.slice(4096 + 64 * index..4096 + 64 * index + 32).unwrap(), Write);
        let wakes = Arc::new(Wakes::default());
        // Fragments, regions of other work admitted before them, fragments of the transfer's other slot.
        let cases = [(1, 0, 0), (64, 0, 64), (64, 3, 0), (256, 3, 64)];
        let comparisons = cases.map(|(fragments, earlier, slot)| {
            let held = (0..earlier)
                .map(|index| match index {
                    0 => {
                        let mut lease = Lease::new(1);
                        lease
                            .enqueue(&Admission::begin(), [Region::from(other(index))])
                            .unwrap();
                        lease
                    }
                    _ => Lease::acquire([other(index)]).unwrap(),
                })
                .collect::<Vec<_>>();
            let admission = Admission::begin();
            let mut previous = Lease::new(slot);
            previous
                .enqueue(&admission, (0..slot).map(|index| fragment(256 + index)))
                .unwrap();
            let mut lease = Lease::new(fragments);
            lease.enqueue(&admission, (0..fragments).map(fragment)).unwrap();
            drop(admission);
            let later = Lease::acquire([other(earlier)]).unwrap();
            let requests = lease.regions().collect::<Vec<_>>();
            let active = requests[0].allocation.active().lock().unwrap();
            let mut count = 0;
            assert!(!active.blocks(&requests, || count += 1));
            drop(active);
            assert_eq!(poll(&mut lease.ready(), &wakes), Poll::Ready(Ok(())));
            drop((held, previous, later));
            count
        });
        assert_eq!(comparisons, [0, 0, 192, 768]);
    }

    /// A slot waits for the conflicting work admitted before it, launches between its transfer's
    /// slots and earlier transfers included, but not for its transfer's other slots or later work.
    #[test]
    fn waits_for_earlier_work_of_others() {
        let buffer = allocation(512);
        let write = |bytes: Range<usize>| Region::from((buffer.slice(bytes).unwrap(), Write));
        let wakes = Arc::new(Wakes::default());
        let admission = Admission::begin();
        let mut first = Lease::new(2);
        first.enqueue(&admission, [write(0..64), write(256..320)]).unwrap();
        let launch = Lease::acquire([(buffer.slice(128..192).unwrap(), Read)]).unwrap();
        let mut second = Lease::new(2);
        second.enqueue(&admission, [write(128..192), write(0..64)]).unwrap();
        drop(admission);
        let admission = Admission::begin();
        let mut later = Lease::new(2);
        later.enqueue(&admission, [write(256..320), write(0..64)]).unwrap();
        drop(admission);

        assert_eq!(poll(&mut first.ready(), &wakes), Poll::Ready(Ok(())));
        assert!(poll(&mut second.ready(), &wakes).is_pending());
        assert!(poll(&mut later.ready(), &wakes).is_pending());
        let wakeups = woken(&wakes);
        drop(launch);
        assert!(woken(&wakes) > wakeups);
        assert_eq!(poll(&mut second.ready(), &wakes), Poll::Ready(Ok(())));
        drop(first);
        assert!(poll(&mut later.ready(), &wakes).is_pending());
        drop(second);
        assert_eq!(poll(&mut later.ready(), &wakes), Poll::Ready(Ok(())));
    }

    /// Queued leases released in any order leave the rest waited for, and a released lease queues
    /// again behind the work queued meanwhile.
    #[test]
    fn releases_queued_leases() {
        let buffer = allocation(512);
        let write = |bytes: Range<usize>| Region::from((buffer.slice(bytes).unwrap(), Write));
        // Two separate fragments, so each slot links two records.
        let slot = |index: usize| {
            [
                write(128 * index..128 * index + 16),
                write(128 * index + 32..128 * index + 48),
            ]
        };
        let wakes = Arc::new(Wakes::default());
        let admission = Admission::begin();
        let mut slots = [0, 1, 2].map(|index| {
            let mut lease = Lease::new(2);
            lease.enqueue(&admission, slot(index)).unwrap();
            lease
        });
        drop(admission);
        let admission = Admission::begin();
        let mut later = Lease::new(3);
        later
            .enqueue(
                &admission,
                [0, 1, 2].map(|index| write(128 * index + 32..128 * index + 48)),
            )
            .unwrap();
        drop(admission);

        slots[1].clear();
        assert!(poll(&mut later.ready(), &wakes).is_pending());
        slots[0].clear();
        assert!(poll(&mut later.ready(), &wakes).is_pending());
        let admission = Admission::begin();
        slots[0].enqueue(&admission, slot(0)).unwrap();
        drop(admission);
        slots[2].clear();
        assert_eq!(poll(&mut later.ready(), &wakes), Poll::Ready(Ok(())));
        assert!(poll(&mut slots[0].ready(), &wakes).is_pending());
        later.clear();
        assert_eq!(poll(&mut slots[0].ready(), &wakes), Poll::Ready(Ok(())));
        slots[0].clear();
        Lease::acquire([(buffer.slice(..).unwrap(), Write)]).unwrap();
    }

    /// A queued lease without records for all its regions holds none of them.
    #[test]
    fn rolls_back_queued_overflow() {
        let buffers = [allocation(256), allocation(256)];
        let regions = buffers
            .iter()
            .flat_map(|buffer| [0..16, 32..48].map(|bytes| Region::from((buffer.slice(bytes).unwrap(), Write))));
        let mut lease = Lease::new(3);
        assert_eq!(lease.enqueue(&Admission::begin(), regions), Err(Error::Capacity));
        let whole = buffers.each_ref().map(|buffer| (buffer.slice(..).unwrap(), Write));
        Lease::acquire(whole).unwrap();
    }

    /// A queued access waits exactly for the earlier work it races with in its allocation: a shared
    /// byte on a shared chip, with a write on either side.
    #[test]
    fn waits_for_racing_work() {
        let buffer = allocation(256);
        let wakes = Arc::new(Wakes::default());
        // Earlier work's (bytes, chips, access), then the queued one's, and whether it waits.
        for (held, queued, waits) in [
            ((0..64, 1, Write), (0..64, 1, Read), true),
            ((0..64, 1, Read), (32..96, 1, Write), true),
            ((0..64, 1, Read), (0..64, 1, Read), false),
            ((0..64, 1, Write), (63..64, 1, Read), true),
            ((0..64, 1, Write), (64..128, 1, Write), false),
            ((0..64, 3, Write), (0..64, 2, Read), true),
            ((0..64, 2, Write), (0..64, 1, Write), false),
            ((0..64, 1, Write), (32..32, 1, Write), false),
            ((0..64, 1, Write), (0..64, 0, Write), false),
        ] {
            let (bytes, chips, access) = held.clone();
            let _held = Lease::acquire([(view(&buffer, bytes, chips), access)]).unwrap();
            let (bytes, chips, access) = queued.clone();
            let mut lease = Lease::new(1);
            let region = Region::from((view(&buffer, bytes, chips), access));
            lease.enqueue(&Admission::begin(), [region]).unwrap();
            assert_eq!(
                poll(&mut lease.ready(), &wakes).is_pending(),
                waits,
                "{held:?} {queued:?}"
            );
        }

        // Fragments alternating between two allocations at the same offsets wait in each only for
        // that allocation's earlier work.
        let (left, right) = (allocation(256), allocation(256));
        let write = |buffer: &Buffer, bytes: Range<usize>| Region::from((buffer.slice(bytes).unwrap(), Write));
        let earlier = Lease::acquire([(right.slice(64..80).unwrap(), Read)]).unwrap();
        let mut lease = Lease::new(4);
        let fragments = [(&left, 0..16), (&right, 0..16), (&left, 64..80), (&right, 64..80)];
        lease
            .enqueue(
                &Admission::begin(),
                fragments.map(|(buffer, bytes)| write(buffer, bytes)),
            )
            .unwrap();
        let mut later = Lease::new(1);
        later.enqueue(&Admission::begin(), [write(&left, 0..16)]).unwrap();
        assert!(poll(&mut lease.ready(), &wakes).is_pending());
        drop(earlier);
        assert_eq!(poll(&mut lease.ready(), &wakes), Poll::Ready(Ok(())));
        assert!(poll(&mut later.ready(), &wakes).is_pending());
        drop(lease);
        assert_eq!(poll(&mut later.ready(), &wakes), Poll::Ready(Ok(())));
    }

    /// Two batches that queue the same two allocations in opposite orders, at the same time, wait
    /// one for the other and never each for the other.
    #[test]
    fn orders_concurrent_batches() {
        let (left, right) = (allocation(256), allocation(256));
        let region = |buffer: &Buffer| Region::from((buffer.slice(..).unwrap(), Write));
        let batch = |first: &Buffer, second: &Buffer| {
            let admission = Admission::begin();
            let mut lease = Lease::new(2);
            lease.enqueue(&admission, [region(first)]).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
            lease.enqueue(&admission, [region(second)]).unwrap();
            lease
        };
        let (forward, backward) = std::thread::scope(|scope| {
            let forward = scope.spawn(|| batch(&left, &right));
            std::thread::sleep(std::time::Duration::from_millis(5));
            let backward = scope.spawn(|| batch(&right, &left));
            (forward.join().unwrap(), backward.join().unwrap())
        });
        let wakes = Arc::new(Wakes::default());
        let (mut forward_ready, mut backward_ready) = (forward.ready(), backward.ready());
        let ready = [poll(&mut forward_ready, &wakes), poll(&mut backward_ready, &wakes)];
        assert_eq!(ready.iter().filter(|ready| ready.is_ready()).count(), 1, "{ready:?}");
        drop((forward_ready, backward_ready));
        let (earlier, mut later) = if ready[0].is_ready() {
            (forward, backward)
        } else {
            (backward, forward)
        };
        drop(earlier);
        assert_eq!(poll(&mut later.ready(), &wakes), Poll::Ready(Ok(())));
        later.clear();
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
            let space = crate::test::allocator(0..4096, 4096);
            let first = space.alloc(Kind::Dram, 512).unwrap();
            let second = space.alloc(Kind::Dram, 512).unwrap();
            let (first, second) = if first.addr() < second.addr() {
                (&first, &second)
            } else {
                (&second, &first)
            };
            assert_eq!(first.addr() + first.len(), second.addr());
            let _first = Lease::acquire([(first.slice(..).unwrap(), Write)]).unwrap();
            assert!(Lease::acquire([(second.slice(..).unwrap(), Write)]).is_ok());
        }

        // A launch holding adjacent blocks: a transfer into a covered block is Busy, one into the
        // gap between blocks is admitted, and the launch's own overlapping alias of the blocks is Busy.
        let buffer = allocation(4096);
        for (blocks, transfer, admitted) in [
            ([0..512, 512..1024], 700..800, false),
            ([0..512, 1024..1536], 512..1024, true),
        ] {
            let behind = blocks.map(|block| buffer.alias(block).unwrap());
            let _launch = Lease::acquire(behind.iter().map(|buffer| (buffer.slice(..).unwrap(), Write))).unwrap();
            let transfer = Lease::acquire([(buffer.slice(transfer).unwrap(), Write)]);
            assert_eq!(transfer.err(), (!admitted).then_some(Error::Busy));
        }
        let behind = [
            buffer.alias(..512).unwrap(),
            buffer.alias(512..1024).unwrap(),
            buffer.alias(256..768).unwrap(),
        ];
        let launch = Lease::acquire(behind.iter().map(|buffer| (buffer.slice(..).unwrap(), Write)));
        assert!(matches!(launch, Err(Error::Busy)));
    }

    /// Runs name every block they cover, across word edges, and nothing else; a difference keeps the
    /// blocks the other set lacks; runs past the blocks and sets of other geometry are refused.
    #[test]
    fn blocks_cover_their_runs() {
        let runs = [0..1, 3..5, 60..70, 127..129, 190..200, 9..9];
        let blocks = Blocks::new(64, 200, runs.clone()).unwrap();
        assert!((0..200).all(|block| blocks.contains(block) == runs.iter().any(|run| run.contains(&block))));
        assert!(matches!(
            Blocks::new(64, 200, std::iter::once(199..201)),
            Err(Error::Invalid(_))
        ));
        assert!(Blocks::new(64, 200, std::iter::once(250..250)).is_ok());
        assert!(matches!(Blocks::new(0, 200, []), Err(Error::Invalid(_))));

        let other = Blocks::new(64, 200, [4..64, 195..196]).unwrap();
        assert_eq!(
            blocks.difference(&other).unwrap(),
            Blocks::new(64, 200, [0..1, 3..4, 64..70, 127..129, 190..195, 196..200]).unwrap()
        );
        for geometry in [Blocks::new(64, 199, []), Blocks::new(32, 200, [])] {
            assert!(matches!(blocks.difference(&geometry.unwrap()), Err(Error::Invalid(_))));
        }
    }

    /// A lease holding some blocks of a view excludes, by each one's access, only accesses reaching
    /// them, admitted at once or queued; its own holds of disjoint blocks do not conflict.
    #[test]
    fn holds_only_named_blocks() {
        let buffer = allocation(1024);
        let span = buffer.slice(..512).unwrap();
        let written = Blocks::new(64, 8, [1..2, 3..4]).unwrap();
        let read = Blocks::new(64, 8, std::iter::once(5..7)).unwrap();
        // The view after the span, with the same access, which a hold of blocks must not absorb.
        let after = (buffer.slice(512..576).unwrap(), Write).into();
        let _held = Lease::acquire([
            Hold::blocks(span, Write, &written).unwrap(),
            after,
            Hold::blocks(span, Read, &read).unwrap(),
        ])
        .unwrap();
        let wakes = Arc::new(Wakes::default());
        for (bytes, access, conflict) in [
            (0..64, Write, false),
            (128..192, Write, false),
            (320..384, Read, false),
            (448..512, Write, false),
            (576..1024, Write, false),
            (0..65, Read, true),
            (200..201, Read, true),
            (256..321, Write, true),
            (384..448, Write, true),
            (512..513, Read, true),
        ] {
            let view = buffer.slice(bytes.clone()).unwrap();
            let refused = if conflict { Err(Error::Busy) } else { Ok(()) };
            assert_eq!(
                Lease::new(1).extend([(view, access)]),
                refused,
                "at once: {bytes:?} {access:?}"
            );
            assert_eq!(
                poll(&mut view.ready(access), &wakes).is_pending(),
                conflict,
                "{bytes:?} {access:?}"
            );
            let mut queued = Lease::new(1);
            queued
                .enqueue(&Admission::begin(), [Region::from((view, access))])
                .unwrap();
            assert_eq!(
                poll(&mut queued.ready(), &wakes).is_pending(),
                conflict,
                "queued: {bytes:?}"
            );
        }

        // Holds of blocks compare block by block, within one lease and across leases.
        let other = allocation(512);
        let span = other.slice(..).unwrap();
        let [first, second] = [1..2, 2..3].map(|run| Blocks::new(64, 8, [run]).unwrap());
        let same = [
            Hold::blocks(span, Write, &first).unwrap(),
            Hold::blocks(span, Read, &first).unwrap(),
        ];
        assert_eq!(Lease::acquire(same).err(), Some(Error::Busy));
        let _first = Lease::acquire([Hold::blocks(span, Write, &first).unwrap()]).unwrap();
        assert!(Lease::acquire([Hold::blocks(span, Write, &second).unwrap()]).is_ok());
        assert_eq!(
            Lease::acquire([Hold::blocks(span, Read, &first).unwrap()]).err(),
            Some(Error::Busy)
        );
        assert!(matches!(
            Hold::blocks(span, Read, &Blocks::new(64, 4, []).unwrap()),
            Err(Error::Invalid(_))
        ));
    }

    /// Two holds of blocks race exactly where a member of each covers a common byte, whatever the
    /// offset between their views and whether their blocks have one size on one grid.
    #[test]
    fn masked_holds_race_where_members_share_a_byte() {
        let len = 8 * 640;
        let buffer = allocation(len);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            usize::try_from(state % u64::try_from(bound).unwrap()).unwrap()
        };
        let trials = 2000;
        let races = (0..trials)
            .filter(|trial| {
                // Some pairs differ in block size, or shift one view by part of a block.
                let [first, second] = [0, 1].map(|side| {
                    let size = if side == 1 && trial % 8 == 5 { 16 } else { 8 };
                    let skew = if side == 1 && trial % 4 == 3 { 1 + next(7) } else { 0 };
                    let count = 1 + next(200);
                    let start = next((len - count * size - 8) / 8) * 8 + skew;
                    let density = 1 + next(120);
                    let members = (0..count).filter(|_| next(400) < density).collect::<Vec<_>>();
                    let blocks = Blocks::new(size, count, members.iter().map(|&block| block..block + 1)).unwrap();
                    let bytes = members
                        .iter()
                        .map(|block| start + block * size..start + (block + 1) * size);
                    (start, blocks, bytes.collect::<Vec<_>>())
                });
                let race = (first.2.iter())
                    .any(|left| (second.2.iter()).any(|right| left.start < right.end && right.start < left.end));
                let [left, right] = [&first, &second]
                    .map(|(start, blocks, _)| buffer.slice(*start..*start + blocks.count * blocks.size).unwrap());
                let holds = [
                    Hold::blocks(left, Write, &first.1).unwrap(),
                    Hold::blocks(right, Read, &second.1).unwrap(),
                ];
                assert_eq!(Lease::acquire(holds).is_err(), race, "one lease: {first:?} {second:?}");
                let held = Lease::acquire([holds[0]]).unwrap();
                assert_eq!(
                    Lease::acquire([holds[1]]).is_err(),
                    race,
                    "two leases: {first:?} {second:?}"
                );
                let mut queued = Lease::new(1);
                queued.enqueue(&Admission::begin(), [Region::from(holds[1])]).unwrap();
                let wakes = Arc::new(Wakes::default());
                assert_eq!(poll(&mut queued.ready(), &wakes).is_pending(), race, "queued");
                drop((queued, held));
                race
            })
            .count();
        assert!(
            (trials / 10..trials * 9 / 10).contains(&races),
            "{races} races in {trials} trials"
        );
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
        next.extend(std::iter::empty::<Region>()).unwrap();
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
                Lease::new(1)
                    .extend([Region::host(host.slice(..).pinned.unwrap(), host.len(), Access::Read)])
                    .err(),
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
