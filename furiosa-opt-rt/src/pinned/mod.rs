//! Pinned host storage and borrowed transfer views.

// allow: SIZE_OK — CPU waiting and DMA leases share allocation-wide completion and poisoning state.

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

/// Set in `DmaBuf::active` once a device access to the memory failed.
const FAILED: usize = 1 << (usize::BITS - 1);
/// The weight a device access that writes the memory adds to `DmaBuf::active`.
const WRITER: usize = 1 << (usize::BITS - 2);
const ORDERED: usize = 1 << (usize::BITS - 3);

/// Page-locked host bytes.
#[derive(Debug)]
pub struct Pinned {
    /// CPU access to the dma-buf; leases retain the buffer itself, not this mapping.
    mmap: MmapMut,
    pub(crate) allocation: Arc<DmaBuf>,
    /// The device DMA mapping [`crate::Device::map`] made.
    pub(crate) mapping: Option<Arc<Mapping>>,
    /// Bytes the caller asked for; the mapping is page-rounded past them.
    pub(crate) bytes: usize,
}

/// The dma-buf behind a pinned range: its descriptor, its device mapping and where the range
/// starts in it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Backing<'a> {
    allocation: &'a Arc<DmaBuf>,
    pub(crate) mapping: Option<&'a Arc<Mapping>>,
    pub(crate) offset: usize,
}

#[derive(Debug)]
pub(crate) struct DmaBuf {
    pub(crate) fd: OwnedFd,
    active: AtomicUsize,
    ordered: Mutex<Option<Arc<Mapping>>>,
    /// Wakes CPU accesses in [`DmaBuf::wait`] once no device access holds the memory.
    idle: (Mutex<()>, Condvar),
}

impl DmaBuf {
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

/// One device access's hold on a pinned allocation, released at its physical completion.
#[derive(Debug)]
pub(crate) struct HostLease {
    allocation: Arc<DmaBuf>,
    weight: usize,
    ordered: bool,
}

impl HostLease {
    pub(crate) fn belongs_to(&self, backing: &Backing<'_>) -> bool {
        Arc::ptr_eq(&self.allocation, backing.allocation)
    }

    pub(crate) fn fd(&self) -> RawFd {
        self.allocation.fd.as_raw_fd()
    }

    pub(crate) fn poison(&self) {
        self.allocation.active.fetch_or(FAILED, Ordering::Release);
        self.allocation.notify();
    }
}

/// The descriptor tables through which a device's clusters reach mapped host memory.
#[derive(Debug)]
pub(crate) struct Mapping {
    pub(crate) table: Buffer,
    /// Where each cluster's table starts in `table`.
    pub(crate) offsets: Vec<usize>,
}

impl Mapping {
    /// The DMA operand cluster `cluster` decodes for `offset` bytes into the mapped memory, on a
    /// chip whose memory its clusters see at `window`.
    pub(crate) fn operand(&self, window: u64, cluster: usize, offset: u32) -> u64 {
        let table = rngd::DEVICE_MEMORY.start + self.table.operand(window) + self.offsets[cluster] as u64;
        rngd::dma::Operand::new(table, offset)
            .expect("registered host descriptor fits the device address space")
            .encode()
    }
}

impl Backing<'_> {
    /// Holds the allocation for a device access; [`Error::Busy`] while another access writes it,
    /// or holds it at all when this one writes.
    pub(crate) fn lease(&self, access: Access) -> Result<HostLease> {
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
        Ok(HostLease {
            allocation: Arc::clone(self.allocation),
            weight,
            ordered: false,
        })
    }

    pub(crate) fn ordered(&self) -> Result<HostLease> {
        let mapping = self.mapping.expect("ordered host operands are mapped");
        let mut domain = self.allocation.ordered.lock().unwrap_or_else(PoisonError::into_inner);
        self.allocation
            .active
            .fetch_update(Ordering::Acquire, Ordering::Acquire, |active| match active {
                0 => Some(ORDERED | 1),
                _ if active & (FAILED | WRITER) == 0
                    && active & ORDERED != 0
                    && domain.as_ref().is_some_and(|other| Arc::ptr_eq(mapping, other)) =>
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
        domain.get_or_insert_with(|| Arc::clone(mapping));
        Ok(HostLease {
            allocation: Arc::clone(self.allocation),
            weight: 1,
            ordered: true,
        })
    }

    /// This backing for the piece at address `piece` of the bytes at `bytes` it backs.
    fn piece(self, bytes: usize, piece: usize) -> Self {
        Self {
            offset: self.offset + (piece - bytes),
            ..self
        }
    }

    /// Whether both ranges lie in one allocation.
    pub(crate) fn shares(&self, other: &Backing<'_>) -> bool {
        Arc::ptr_eq(self.allocation, other.allocation)
    }

    /// Another hold for an access that already leased this allocation.
    pub(crate) fn retain(&self) -> HostLease {
        self.allocation.active.fetch_add(1, Ordering::Relaxed);
        HostLease {
            allocation: Arc::clone(self.allocation),
            weight: 1,
            ordered: false,
        }
    }
}

impl Drop for HostLease {
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

impl Pinned {
    /// Number of bytes; querying metadata does not wait for DMA.
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Always false: [`Pinned::zeroed`] refuses empty memory.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

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
            mapping: None,
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
                mapping: self.mapping.as_ref(),
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
                mapping: self.mapping.as_ref(),
                offset,
            }),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::pinned;

    /// Empty pinned memory is the caller's invalid request.
    #[test]
    fn refuses_empty_memory() {
        assert!(matches!(Pinned::zeroed(0), Err(Error::Invalid(_))));
    }

    /// CPU access sleeps while a device access holds the memory, and wakes when it ends.
    #[test]
    fn waits_without_spinning() {
        use nix::time::{ClockId, clock_gettime};
        use std::time::Duration;
        let mut host = pinned();
        let lease = host.slice(..).backing.unwrap().lease(Access::Write).unwrap();
        let reader = std::thread::spawn(move || {
            let cpu = || Duration::from(clock_gettime(ClockId::CLOCK_THREAD_CPUTIME_ID).unwrap());
            let start = cpu();
            host[0] = 1;
            cpu() - start
        });
        std::thread::sleep(Duration::from_millis(200));
        drop(lease);
        let busy = reader.join().unwrap();
        assert!(busy < Duration::from_millis(50), "waited on the CPU for {busy:?}");
    }
}
