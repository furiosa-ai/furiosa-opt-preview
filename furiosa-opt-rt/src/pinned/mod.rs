//! Pinned host storage and borrowed transfer views.

mod view;
pub use view::{HostBufferSlice, HostBufferSliceMut};

use std::ops::{Deref, DerefMut};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::slice::SliceIndex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use dma_heap::{Heap, HeapKind};
use memmap2::MmapMut;

use crate::arch::rngd;
use crate::{Access, Buffer, Error, Result};

/// Page-locked host bytes.
#[derive(Debug)]
pub struct Pinned {
    /// CPU access to the dma-buf; leases retain the buffer itself, not this mapping.
    mmap: MmapMut,
    allocation: Arc<DmaBuf>,
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
        let mmap = unsafe { MmapMut::map_mut(&fd) }
            .map_err(|why| Error::Device(format!("mapping {bytes} bytes of pinned memory: {why}")))?;
        Ok(Self {
            mmap,
            allocation: Arc::new(DmaBuf {
                fd,
                active: AtomicUsize::new(0),
                ordered: Mutex::new(None),
                idle: Default::default(),
            }),
            registration: None,
            bytes,
        })
    }

    /// Borrows a range of bytes without losing the DMA-buf or its byte offset.
    pub fn slice<I>(&self, range: I) -> HostBufferSlice<'_>
    where
        I: SliceIndex<[u8], Output = [u8]>,
    {
        let bytes = &self.mmap[..self.bytes][range];
        HostBufferSlice {
            bytes,
            backing: Some(Backing {
                allocation: &self.allocation,
                registration: self.registration.as_ref(),
                offset: bytes.as_ptr().addr() - self.mmap.as_ptr().addr(),
            }),
        }
    }

    /// Exclusively borrows a range of bytes for a read from the device.
    pub fn slice_mut<I>(&mut self, range: I) -> HostBufferSliceMut<'_>
    where
        I: SliceIndex<[u8], Output = [u8]>,
    {
        let base = self.mmap.as_ptr().addr();
        let bytes = &mut self.mmap[..self.bytes][range];
        let offset = bytes.as_ptr().addr() - base;
        HostBufferSliceMut {
            bytes,
            backing: Some(Backing {
                allocation: &self.allocation,
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

    /// The dma-buf behind the memory.
    pub(crate) fn allocation(&self) -> &Arc<DmaBuf> {
        &self.allocation
    }

    /// The registration [`crate::Device::map`] made, if it made one.
    pub(crate) fn registration(&self) -> Option<&Arc<Registration>> {
        self.registration.as_ref()
    }

    /// Makes `registration` the one the memory's device DMA goes through.
    pub(crate) fn set_registration(&mut self, registration: Option<Arc<Registration>>) {
        self.registration = registration;
    }
}

/// CPU access blocks while device work holds the memory, and panics once a device access to it
/// failed, since its contents are then unknown.
impl Deref for Pinned {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.allocation.wait();
        &self.mmap[..self.bytes]
    }
}

/// CPU access blocks while device work holds the memory, and panics once a device access to it
/// failed, since its contents are then unknown.
impl DerefMut for Pinned {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.allocation.wait();
        &mut self.mmap[..self.bytes]
    }
}

/// The dma-buf behind a pinned range: its descriptor, its device registration and where the range
/// starts in it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Backing<'a> {
    allocation: &'a Arc<DmaBuf>,
    registration: Option<&'a Arc<Registration>>,
    offset: usize,
}

impl<'a> Backing<'a> {
    /// Holds the allocation for a device access; [`Error::Busy`] while another access writes it,
    /// or holds it at all when this one writes.
    pub(crate) fn lease(&self, access: Access) -> Result<PinnedLease> {
        let weight = match access {
            Access::Read => 1,
            Access::Write => WRITER,
        };
        self.allocation
            .active
            .fetch_update(Ordering::Acquire, Ordering::Acquire, |active| match active {
                0 => Some(weight),
                _ if access == Access::Read && active < ORDERED => Some(active + 1),
                _ => None,
            })
            .map_err(|active| {
                if active & FAILED == 0 {
                    Error::Busy
                } else {
                    Error::Poisoned
                }
            })?;
        Ok(PinnedLease {
            allocation: Arc::clone(self.allocation),
            weight,
            ordered: false,
            offset: 0,
        })
    }

    pub(crate) fn ordered(&self) -> Result<PinnedLease> {
        let registration = self.registration.expect("ordered host operands are mapped");
        let mut domain = self.allocation.ordered.lock().unwrap_or_else(PoisonError::into_inner);
        self.allocation
            .active
            .fetch_update(Ordering::Acquire, Ordering::Acquire, |active| match active {
                0 => Some(ORDERED | 1),
                _ if active & (FAILED | WRITER) == 0
                    && active & ORDERED != 0
                    && domain.as_ref().is_some_and(|other| Arc::ptr_eq(registration, other)) =>
                {
                    Some(active + 1)
                }
                _ => None,
            })
            .map_err(|active| {
                if active & FAILED == 0 {
                    Error::Busy
                } else {
                    Error::Poisoned
                }
            })?;
        domain.get_or_insert_with(|| Arc::clone(registration));
        Ok(PinnedLease {
            allocation: Arc::clone(self.allocation),
            weight: 1,
            ordered: true,
            offset: 0,
        })
    }

    /// Another hold for an access that already leased this allocation.
    pub(crate) fn retain(&self) -> PinnedLease {
        self.allocation.active.fetch_add(1, Ordering::Relaxed);
        PinnedLease {
            allocation: Arc::clone(self.allocation),
            weight: 1,
            ordered: false,
            offset: 0,
        }
    }

    /// Whether both ranges lie in one allocation.
    pub(crate) fn shares(&self, other: &Backing<'_>) -> bool {
        Arc::ptr_eq(self.allocation, other.allocation)
    }

    /// The registration through which a device's clusters reach this memory, if it has one.
    pub(crate) fn registration(&self) -> Option<&'a Arc<Registration>> {
        self.registration
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

/// One device access's hold on a pinned allocation, released at its physical completion.
#[derive(Debug)]
pub(crate) struct PinnedLease {
    allocation: Arc<DmaBuf>,
    weight: usize,
    ordered: bool,
    /// Where the bytes the access moves start in the allocation.
    offset: usize,
}

impl PinnedLease {
    /// The lease for the access's bytes starting at `offset` of the allocation.
    pub(crate) fn at(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    pub(crate) fn poison(&self) {
        self.allocation.active.fetch_or(FAILED, Ordering::Release);
        self.allocation.notify();
    }

    pub(crate) fn belongs_to(&self, backing: &Backing<'_>) -> bool {
        Arc::ptr_eq(&self.allocation, backing.allocation)
    }

    /// Where the bytes the access moves start in the allocation.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn fd(&self) -> RawFd {
        self.allocation.fd.as_raw_fd()
    }
}

impl Drop for PinnedLease {
    fn drop(&mut self) {
        let active = if self.ordered {
            let mut domain = self.allocation.ordered.lock().unwrap_or_else(PoisonError::into_inner);
            let previous = self
                .allocation
                .active
                .fetch_update(Ordering::Release, Ordering::Relaxed, |active| {
                    Some(if active & !(FAILED | ORDERED) == 1 {
                        active & FAILED
                    } else {
                        active - 1
                    })
                })
                .expect("ordered lease release always succeeds");
            if previous & !(FAILED | ORDERED) == 1 {
                *domain = None;
                previous & FAILED
            } else {
                previous - 1
            }
        } else {
            self.allocation.active.fetch_sub(self.weight, Ordering::Release) - self.weight
        };
        if active & !FAILED == 0 {
            self.allocation.notify();
        }
    }
}

/// One pinned allocation registered with a device: each cluster's segment table, in one allocation
/// of device memory.
#[derive(Debug)]
pub(crate) struct Registration {
    pub(crate) buffer: Buffer,
    /// Where each cluster's table starts in `buffer`, by cluster.
    pub(crate) tables: Vec<usize>,
}

impl Registration {
    /// The DMA operand cluster `cluster` decodes for `offset` bytes into the mapped memory, on a
    /// chip whose memory its clusters see at `window`.
    pub(crate) fn operand(&self, window: u64, cluster: usize, offset: u32) -> u64 {
        let table = rngd::DEVICE_MEMORY.start + self.buffer.operand(window) + self.tables[cluster] as u64;
        rngd::dma::Operand::new(table, offset)
            .expect("registered host descriptor fits the device address space")
            .encode()
    }
}

#[derive(Debug)]
pub(crate) struct DmaBuf {
    fd: OwnedFd,
    active: AtomicUsize,
    ordered: Mutex<Option<Arc<Registration>>>,
    /// Wakes CPU accesses in [`DmaBuf::wait`] once no device access holds the memory.
    idle: (Mutex<()>, Condvar),
}

impl DmaBuf {
    pub(crate) fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Sleeps until no device access holds the memory; panics once one failed.
    fn wait(&self) {
        let idle = || {
            let active = self.active.load(Ordering::Acquire);
            assert_eq!(
                active & FAILED,
                0,
                "DMA failed; this host allocation is no longer accessible"
            );
            active == 0
        };
        if idle() {
            return;
        }
        // Checked again under the lock, which a lease takes before it notifies.
        let (lock, wake) = &self.idle;
        let guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        drop(
            wake.wait_while(guard, |()| !idle())
                .unwrap_or_else(PoisonError::into_inner),
        );
    }

    /// Wakes [`DmaBuf::wait`] after a change of `active`; taking the lock orders the wake
    /// after a waiter's check.
    fn notify(&self) {
        let (lock, idle) = &self.idle;
        drop(lock.lock().unwrap_or_else(PoisonError::into_inner));
        idle.notify_all();
    }
}

/// Set in `DmaBuf::active` once a device access to the memory failed.
const FAILED: usize = 1 << (usize::BITS - 1);
/// The weight a device access that writes the memory adds to `DmaBuf::active`.
const WRITER: usize = 1 << (usize::BITS - 2);
const ORDERED: usize = 1 << (usize::BITS - 3);

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty pinned memory is the caller's invalid request.
    #[test]
    fn refuses_empty_memory() {
        assert!(matches!(Pinned::zeroed(0), Err(Error::Invalid(_))));
    }
}
