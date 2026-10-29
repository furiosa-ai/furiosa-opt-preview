//! Pinned host storage and borrowed transfer views.

mod view;
pub use view::{Bytes, BytesMut, HostBufferSlice, HostBufferSliceMut};

use std::ops::RangeBounds;
use std::ops::{Deref, DerefMut};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex};

use dma_heap::{Heap, HeapKind};
use memmap2::MmapRaw;

use crate::arch::rngd;
use crate::buffer::{Active, Allocator, Request};
use crate::{Access, Buffer, Error, Result};

/// Page-locked host bytes.
pub struct Pinned {
    /// CPU access to the dma-buf; leases retain the buffer itself, not this mapping.
    mmap: MmapRaw,
    dmabuf: Arc<DmaBuf>,
    /// The registration [`crate::Device::map`] made.
    registration: Option<Arc<Registration>>,
    /// Bytes the caller asked for; the mapping is page-rounded past them.
    bytes: usize,
}

impl Pinned {
    /// `len` zeroed bytes of page-locked memory; [`Error::Invalid`] for empty memory.
    pub fn zeroed(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::Invalid("empty memory cannot be pinned".into()));
        }
        let fd = Heap::new(HeapKind::System)
            .and_then(|heap| heap.allocate(len))
            .map_err(|why| Error::Device(format!("allocating {len} bytes of pinned memory: {why}")))?;
        Self::new(fd, len)
    }

    pub(crate) fn new(fd: OwnedFd, bytes: usize) -> Result<Self> {
        // SAFETY: this allocation owns the file and its only mapping.
        let mmap = MmapRaw::map_raw(&fd)
            .map_err(|why| Error::Device(format!("mapping {bytes} bytes of pinned memory: {why}")))?;
        Ok(Self {
            mmap,
            dmabuf: Arc::new(DmaBuf {
                fd,
                active: Mutex::default(),
            }),
            registration: None,
            bytes,
        })
    }

    /// Borrows a range of bytes without losing the DMA-buf or its byte offset.
    pub fn slice<I>(&self, range: I) -> HostBufferSlice<'_>
    where
        I: RangeBounds<usize>,
    {
        let range = self.range(range);
        let address = self.mmap.as_ptr().expose_provenance() + range.start;
        HostBufferSlice {
            address,
            len: range.len(),
            borrow: std::marker::PhantomData,
            pinned: Some(Backing {
                dmabuf: &self.dmabuf,
                registration: self.registration.as_ref(),
                offset: range.start,
            }),
        }
    }

    /// Exclusively borrows a range of bytes for a read from the device.
    pub fn slice_mut<I>(&mut self, range: I) -> HostBufferSliceMut<'_>
    where
        I: RangeBounds<usize>,
    {
        let range = self.range(range);
        let offset = range.start;
        let address = self.mmap.as_mut_ptr().expose_provenance() + offset;
        HostBufferSliceMut {
            address,
            len: range.len(),
            borrow: std::marker::PhantomData,
            pinned: Some(Backing {
                dmabuf: &self.dmabuf,
                registration: self.registration.as_ref(),
                offset,
            }),
        }
    }

    /// Number of bytes; querying metadata does not wait for DMA.
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Always false: [`Pinned::zeroed`] refuses empty memory.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    fn range(&self, range: impl RangeBounds<usize>) -> std::ops::Range<usize> {
        use std::ops::Bound;
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start.checked_add(1).expect("range start overflow"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end.checked_add(1).expect("range end overflow"),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.bytes,
        };
        assert!(start <= end && end <= self.bytes, "pinned range exceeds allocation");
        start..end
    }

    /// The dma-buf behind the memory.
    pub(crate) fn dmabuf(&self) -> &Arc<DmaBuf> {
        &self.dmabuf
    }

    /// The registration [`crate::Device::map`] made, if it made one.
    pub(crate) fn registration(&self) -> Option<&Arc<Registration>> {
        self.registration.as_ref()
    }

    /// All of the memory, as a CPU `access` to it.
    fn request(&self, access: Access) -> Request<'_> {
        let backing = Backing {
            dmabuf: &self.dmabuf,
            registration: None,
            offset: 0,
        };
        Request::host(backing, self.bytes, access)
    }

    /// Makes `registration` the one the memory's device DMA goes through.
    pub(crate) fn set_registration(&mut self, registration: Option<Arc<Registration>>) {
        self.registration = registration;
    }
}

/// CPU access blocks while device work holds the memory, and panics once a device access to it
/// failed, since its contents are then unknown; a view's `bytes` reports that instead.
impl Deref for Pinned {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.request(Access::Read)
            .wait()
            .expect("DMA failed; this host allocation is no longer accessible");
        // SAFETY: earlier device writes ended, and the returned borrow keeps later ones off the bytes.
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr(), self.bytes) }
    }
}

/// CPU access blocks while device work holds the memory, and panics once a device access to it
/// failed, since its contents are then unknown; a view's `into_bytes` reports that instead.
impl DerefMut for Pinned {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.request(Access::Write)
            .wait()
            .expect("DMA failed; this host allocation is no longer accessible");
        // SAFETY: earlier device accesses ended, and this exclusive borrow keeps later ones off the bytes.
        unsafe { std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr(), self.bytes) }
    }
}

/// The dma-buf behind a pinned range: its descriptor, its device registration and where the range
/// starts in it.
#[derive(Clone, Copy)]
pub(crate) struct Backing<'a> {
    pub(crate) dmabuf: &'a Arc<DmaBuf>,
    pub(crate) registration: Option<&'a Arc<Registration>>,
    pub(crate) offset: usize,
}

impl<'a> Backing<'a> {
    /// The registration through which `allocator`'s device reaches this memory, if `map` made one
    /// there.
    pub(crate) fn mapped_to(&self, allocator: &Arc<Allocator>) -> Option<&'a Arc<Registration>> {
        self.registration
            .filter(|registration| registration.belongs_to(allocator))
    }

    /// Where the range starts in its dma-buf.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    /// This backing for the piece at address `piece` of the bytes at `bytes` it backs.
    fn piece(self, bytes: usize, piece: usize) -> Self {
        Self {
            offset: self.offset + (piece - bytes),
            ..self
        }
    }
}

impl std::fmt::Debug for Pinned {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Pinned").field("bytes", &&**self).finish()
    }
}

/// One pinned allocation registered with a device: each cluster's segment table, laid out in
/// device DRAM.
#[derive(Debug)]
pub(crate) struct Registration {
    /// The device DRAM the tables lie in.
    pub(crate) tables: Buffer,
    /// Where each cluster's table starts in `tables`, by cluster.
    pub(crate) offsets: Vec<usize>,
}

impl Registration {
    /// Whether the registration is `allocator`'s device's: its tables lie in that device's memory.
    pub(crate) fn belongs_to(&self, allocator: &Arc<Allocator>) -> bool {
        self.tables.belongs_to(allocator)
    }

    /// The DMA operand cluster `cluster` decodes for `offset` bytes into the mapped memory, on a
    /// chip whose memory its clusters see at `window`.
    pub(crate) fn operand(&self, window: u64, cluster: usize, offset: u32) -> u64 {
        let table = rngd::DEVICE_MEMORY.start + self.tables.operand(window) + self.offsets[cluster] as u64;
        rngd::dma::Operand::new(table, offset)
            .expect("registered host descriptor fits the device address space")
            .encode()
    }
}

pub(crate) struct DmaBuf {
    pub(crate) fd: OwnedFd,
    pub(crate) active: Mutex<Active>,
}

impl DmaBuf {
    pub(crate) fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::buffer::Lease;

    /// Device work on disjoint pinned ranges does not exclude another range's writer.
    #[test]
    fn admits_disjoint_ranges() {
        let host = crate::test::pinned();
        let mut first = Lease::new(1);
        first
            .extend([Request::host(host.slice(..128).pinned.unwrap(), 128, Access::Write)])
            .unwrap();
        let mut second = Lease::new(1);
        let second = second.extend([Request::host(host.slice(128..256).pinned.unwrap(), 128, Access::Write)]);
        assert!(second.is_ok());
        drop(first);
    }

    /// Poisoned transfer views refuse CPU reads and writes with `Poisoned`, exposing no bytes.
    #[test]
    fn refuses_poisoned_views() {
        let mut host = crate::test::pinned();
        let mut lease = Lease::new(1);
        lease
            .extend([Request::host(host.slice(..).pinned.unwrap(), host.len(), Access::Write)])
            .unwrap();
        lease.poison();
        drop(lease);
        assert!(matches!(host.slice(..).bytes(), Err(Error::Poisoned)));
        assert!(matches!(host.slice_mut(..).into_bytes(), Err(Error::Poisoned)));
    }

    /// A CPU read waits only for the device writes of its own bytes.
    #[test]
    fn reads_beside_device_writes() {
        let host = crate::test::pinned();
        let (head, tail) = (host.slice(..128), host.slice(128..));
        let mut writer = Lease::new(1);
        writer
            .extend([Request::host(tail.pinned.unwrap(), tail.len(), Access::Write)])
            .unwrap();
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| head.bytes().map(|bytes| bytes[0]));
            std::thread::sleep(Duration::from_millis(50));
            let finished = reader.is_finished();
            drop(writer);
            assert!(finished, "a CPU read waited for a device write of other bytes");
            assert_eq!(reader.join().unwrap(), Ok(0));
        });
    }

    /// CPU readers coexist, and each waits for the device writes admitted before it.
    #[test]
    fn shares_cpu_reads() {
        let host = crate::test::pinned();
        let view = host.slice(..);
        let mut writer = Lease::new(1);
        writer
            .extend([Request::host(view.pinned.unwrap(), view.len(), Access::Write)])
            .unwrap();
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| view.bytes().map(|bytes| bytes[0]));
            std::thread::sleep(Duration::from_millis(50));
            assert!(!reader.is_finished(), "a CPU read waits for the device write");
            drop(writer);
            assert_eq!(reader.join().unwrap(), Ok(0));
        });
        let (first, second) = (view.bytes().unwrap(), view.bytes().unwrap());
        assert_eq!(first[0], second[0]);
    }

    /// A split device write excludes other accesses until its last command ends.
    #[test]
    fn excludes_reads_until_last_write() {
        let host = crate::test::pinned();
        let backing = host.slice(..).pinned.unwrap();
        let mut first = Lease::new(1);
        first
            .extend([Request::host(backing, host.len(), Access::Write)])
            .unwrap();
        let last = first;
        assert!(matches!(
            Lease::new(1).extend([Request::host(backing, host.len(), Access::Read)]),
            Err(Error::Busy)
        ));
        drop(last);
        assert!(
            Lease::new(1)
                .extend([Request::host(backing, host.len(), Access::Read)])
                .is_ok()
        );
    }

    /// Retaining an ordered hold leaves the allocation accessible after both holds end.
    #[test]
    fn releases_retained_ordered_memory() {
        let device = crate::test::device(&[0], 8);
        let host = crate::test::mapped(&device, vec![0], 4096);
        let backing = host.slice(..).pinned.unwrap();
        let mut first = Lease::new(1);
        first.set_owner(crate::buffer::Owner::Ordered(1));
        first
            .extend([Request::host(backing, host.len(), Access::Write)])
            .unwrap();
        let mut last = Lease::new(1);
        last.set_owner(crate::buffer::Owner::Ordered(2));
        last.extend([Request::host(backing, host.len(), Access::Write)])
            .unwrap();
        drop(first);
        drop(last);
        assert!(
            Lease::new(1)
                .extend([Request::host(backing, host.len(), Access::Write)])
                .is_ok()
        );
    }

    /// Debugging poisoned pinned memory must not expose its bytes.
    #[test]
    fn refuses_poisoned_debug() {
        let host = crate::test::pinned();
        let mut lease = Lease::new(1);
        lease
            .extend([Request::host(host.slice(..).pinned.unwrap(), host.len(), Access::Write)])
            .unwrap();
        lease.poison();
        drop(lease);
        assert!(std::panic::catch_unwind(|| format!("{host:?}")).is_err());
    }

    /// Each half of a split keeps its place in the DMA-buf the driver addresses.
    #[test]
    fn splits_keep_their_offsets() {
        let mut pinned = crate::test::pinned();
        let (head, tail) = pinned.slice_mut(16..).split_at(32);
        assert_eq!((head.len(), tail.len()), (32, 4096 - 48));
        assert_eq!((head.pinned.unwrap().offset(), tail.pinned.unwrap().offset()), (16, 48));
    }

    /// Empty pinned memory is the caller's invalid request.
    #[test]
    fn refuses_empty_memory() {
        assert!(matches!(Pinned::zeroed(0), Err(Error::Invalid(_))));
    }
}
