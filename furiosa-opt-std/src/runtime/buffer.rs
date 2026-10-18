//! Device memory and the host memory that transfers move it from and to.

use std::fmt::{self, Debug};
use std::ops::{Deref, DerefMut, Range, RangeBounds, RangeFull};
use std::sync::OnceLock;

use furiosa_opt_macro::primitive;

use crate::Error;
use crate::backend::{Allocation, Backend, Runtime};
use crate::runtime::{CurrentBackend, Kind, Usage};

/// A byte range at the same offset on every chip of a device, which an
/// [`HbmTensor`](crate::tensor::memory::HbmTensor) takes by `TryFrom` and gives back by
/// `into_buffer`. [`HbmBuffer::slice`] picks chips and a byte range of each, for a transfer or,
/// through `HbmTensorView::try_from`, a typed tensor view. It splits into disjoint owners with
/// `split_at`, and has no `Clone`; `alias` makes a second owner. Compiled launches reuse chip-local
/// memory in submission order, and a transfer waits for the device work submitted before it on its
/// bytes; a launch that conflicts with multi-chip peer memory, or with a transfer in flight, is
/// `DeviceError::Busy`. `addr()` is where it starts.
#[primitive(HbmBuffer)]
pub struct HbmBuffer<B: Runtime = CurrentBackend>(pub(crate) B::Buffer, OnceLock<B::Buffer>);

mod table;

impl<B: Runtime> HbmBuffer<B> {
    pub(crate) fn new(buffer: B::Buffer) -> Self {
        Self(buffer, OnceLock::new())
    }
}

impl<B: Backend> HbmBuffer<B> {
    /// `bytes` uninitialized bytes on every chip of `device`, which a multi-chip function's device
    /// code reaches only on its own chip; [`DeviceError::OutOfMemory`](super::DeviceError) when no
    /// span is free.
    pub fn alloc(device: &super::Device<B>, bytes: usize) -> Result<Self, Error> {
        B::alloc(device.pdma.device(), Kind::Dram, bytes).map(Self::new)
    }

    /// `bytes` uninitialized bytes on every chip of `device` that the other chips' device code may
    /// write too: a function whose cross-chip transfer lands in a caller's buffer binds only these.
    /// Each chip sees its peers' DRAM through a few 4-GiB windows, so this memory is scarce.
    pub fn alloc_shared(device: &super::Device<B>, bytes: usize) -> Result<Self, Error> {
        B::alloc(device.pdma.device(), Kind::DramShared, bytes).map(Self::new)
    }

    /// Resolves once the launches and transfers submitted before this call that write these bytes
    /// ended, without copying them.
    pub fn ready(&self) -> impl Future<Output = Result<(), Error>> + Send + use<B> {
        self.0.ready()
    }

    /// Bytes on each chip (an `HbmTensor`'s `size()` counts every chip).
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Where the buffer starts in each chip's range.
    pub fn addr(&self) -> usize {
        self.0.addr()
    }

    /// Two owners: every chip's bytes before `offset`, and from it on.
    pub fn split_at(self, offset: usize) -> Result<(Self, Self), Error> {
        if offset > self.size() {
            return Err(Error::Invalid("the offset is outside the buffer"));
        }
        let (head, tail) = self.0.split_at(offset)?;
        Ok((Self::new(head), Self::new(tail)))
    }

    /// Another owner of `bytes` of every chip's range, sharing them without a copy.
    pub fn alias(&self, bytes: impl RangeBounds<usize>) -> Result<Self, Error> {
        let bytes =
            std::slice::try_range(bytes, ..self.size()).ok_or(Error::Invalid("the range is outside the buffer"))?;
        self.0.alias(bytes).map(Self::new)
    }

    /// `bytes` of each chip's range on `chips`, to copy to the host with [`HbmBufferSlice::to_host`].
    pub fn slice(
        &self,
        chips: impl Into<Chips>,
        bytes: impl RangeBounds<usize>,
    ) -> Result<HbmBufferSlice<'_, B>, Error> {
        Ok(HbmBufferSlice {
            region: self.region(chips.into(), bytes)?,
            buffer: &self.0,
            behind: &[],
        })
    }

    /// `bytes` of each chip's range on `chips`, for a copy from the host to `.output(..)` into.
    pub fn slice_mut(
        &mut self,
        chips: impl Into<Chips>,
        bytes: impl RangeBounds<usize>,
    ) -> Result<HbmBufferSliceMut<'_, B>, Error> {
        Ok(HbmBufferSliceMut {
            region: self.region(chips.into(), bytes)?,
            buffer: &mut self.0,
        })
    }

    fn region(&self, chips: Chips, bytes: impl RangeBounds<usize>) -> Result<Region, Error> {
        let chips = chips
            .within(self.0.chips())
            .ok_or(Error::Invalid("the chips are outside the buffer"))?;
        let bytes =
            std::slice::try_range(bytes, ..self.size()).ok_or(Error::Invalid("the range is outside the buffer"))?;
        Ok(Region { chips, bytes })
    }
}

/// A set of a device's chip ranks: `..` for every chip, a range, one rank, or an array of ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chips(Ranks);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ranks {
    All,
    /// One bit per rank; ranks from 63 on share the top bit, which no buffer's chips include.
    // ponytail: one u64 of ranks; widen when a device holds more than 63 chips.
    Mask(u64),
}

impl Chips {
    fn ranks(ranks: impl IntoIterator<Item = usize>) -> Self {
        Self(Ranks::Mask(
            ranks.into_iter().fold(0, |mask, rank| mask | 1 << rank.min(63)),
        ))
    }

    /// These ranks as a mask among a buffer's ranks `0..chips`, or `None` when one is outside.
    fn within(self, chips: usize) -> Option<u64> {
        let all = (1u64 << chips.min(63)) - 1;
        match self.0 {
            Ranks::All => Some(all),
            Ranks::Mask(mask) => (mask & !all == 0).then_some(mask),
        }
    }
}

impl From<RangeFull> for Chips {
    fn from(_: RangeFull) -> Self {
        Self(Ranks::All)
    }
}

impl From<Range<usize>> for Chips {
    fn from(ranks: Range<usize>) -> Self {
        Self::ranks(ranks)
    }
}

impl From<usize> for Chips {
    fn from(rank: usize) -> Self {
        Self::ranks([rank])
    }
}

impl<const N: usize> From<[usize; N]> for Chips {
    fn from(ranks: [usize; N]) -> Self {
        Self::ranks(ranks)
    }
}

/// The chips and per-chip bytes a transfer moves; the host side is those bytes, chip after chip
/// in rank order.
#[derive(Clone, Debug)]
pub(crate) struct Region {
    /// One bit per rank.
    pub(crate) chips: u64,
    pub(crate) bytes: Range<usize>,
}

impl Region {
    pub(super) fn whole(buffer: &impl Allocation) -> Self {
        Self {
            chips: (1 << buffer.chips()) - 1,
            bytes: 0..buffer.size(),
        }
    }

    /// Host bytes this region takes: its bytes on each of its chips.
    pub(crate) fn host_len(&self) -> usize {
        self.bytes.len() * self.chips.count_ones() as usize
    }
}

/// Part of a [`HbmBuffer`] to read: some of its chips and a byte range of each.
#[derive(Debug)]
pub struct HbmBufferSlice<'a, B: Runtime = CurrentBackend> {
    pub(crate) buffer: &'a B::Buffer,
    pub(crate) region: Region,
    /// Memory named by this address table, held with the table's access until completion.
    pub(crate) behind: &'a [B::Buffer],
}

/// Part of a [`HbmBuffer`] to write: some of its chips and a byte range of each.
#[derive(Debug)]
pub struct HbmBufferSliceMut<'a, B: Runtime = CurrentBackend> {
    pub(crate) buffer: &'a mut B::Buffer,
    pub(crate) region: Region,
}

impl<'a, B: Backend> HbmBufferSlice<'a, B> {
    /// Bytes this slice covers on each of its chips.
    pub fn size(&self) -> usize {
        self.region.bytes.len()
    }

    /// Whether the slice covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.region.bytes.is_empty()
    }

    /// Every chip and byte of `buffer`.
    pub(crate) fn whole(buffer: &'a B::Buffer) -> Self {
        Self {
            region: Region::whole(buffer),
            buffer,
            behind: &[],
        }
    }
}

impl<'a, B: Backend> HbmBufferSliceMut<'a, B> {
    /// Every chip and byte of `buffer`.
    pub(crate) fn whole(buffer: &'a mut B::Buffer) -> Self {
        Self {
            region: Region::whole(buffer),
            buffer,
        }
    }

    /// The same bytes, to read.
    pub(crate) fn as_ref(&self) -> HbmBufferSlice<'_, B> {
        HbmBufferSlice {
            buffer: &*self.buffer,
            region: self.region.clone(),
            behind: &[],
        }
    }
}

impl<B: Backend> Clone for HbmBufferSlice<'_, B> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            region: self.region.clone(),
            behind: self.behind,
        }
    }
}

impl<'a, B: Backend> From<&'a HbmBuffer<B>> for HbmBufferSlice<'a, B> {
    fn from(buffer: &'a HbmBuffer<B>) -> Self {
        buffer.slice(.., ..).expect("every chip and byte is inside the buffer")
    }
}

impl<'a, B: Backend> From<&'a mut HbmBuffer<B>> for HbmBufferSliceMut<'a, B> {
    fn from(buffer: &'a mut HbmBuffer<B>) -> Self {
        buffer
            .slice_mut(.., ..)
            .expect("every chip and byte is inside the buffer")
    }
}

/// Host memory for transfers: page-locked where the system provides it, so a transfer reads or
/// writes it directly and asynchronously; otherwise ordinary memory, whose transfer completes before
/// its first poll returns. CPU access through `Deref` waits for any transfer still using it.
///
/// # Panics
/// CPU access panics if failed device work leaves the pinned buffer's contents unknown.
pub struct HostBuffer(HostMemory);

enum HostMemory {
    Pinned(furiosa_opt_rt::Pinned),
    Pageable(Vec<u8>),
}

impl HostBuffer {
    /// `len` zeroed bytes, page-locked where the system provides it.
    pub fn zero(len: usize) -> Self {
        // ponytail: an allocation that cannot be page-locked stays usable, only synchronous.
        Self(
            furiosa_opt_rt::Pinned::zeroed(len).map_or_else(|_| HostMemory::Pageable(vec![0; len]), HostMemory::Pinned),
        )
    }

    /// `bytes` as they are, in ordinary memory.
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        Self(HostMemory::Pageable(bytes))
    }

    /// Bytes of memory. Unlike the slice `Deref` gives, it does not wait for transfers using it.
    pub fn len(&self) -> usize {
        match &self.0 {
            HostMemory::Pinned(pinned) => pinned.len(),
            HostMemory::Pageable(bytes) => bytes.len(),
        }
    }

    /// Whether the memory holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the memory is page-locked, so transfers use it directly and asynchronously.
    pub fn is_pinned(&self) -> bool {
        matches!(self.0, HostMemory::Pinned(_))
    }

    /// The same bytes, moved to page-locked memory; empty or already pinned memory stays.
    pub(crate) fn pin(self) -> Result<Self, Error> {
        match self.0 {
            HostMemory::Pageable(bytes) if !bytes.is_empty() => {
                let mut pinned = furiosa_opt_rt::Pinned::zeroed(bytes.len())?;
                pinned.copy_from_slice(&bytes);
                Ok(Self(HostMemory::Pinned(pinned)))
            }
            memory => Ok(Self(memory)),
        }
    }

    /// `bytes` of this memory, to copy to a device with [`HostBufferSlice::to_hbm`].
    pub fn slice(&self, bytes: impl RangeBounds<usize>) -> Result<HostBufferSlice<'_>, Error> {
        let bytes =
            std::slice::try_range(bytes, ..self.len()).ok_or(Error::Invalid("the range is outside the host buffer"))?;
        Ok(HostBufferSlice(match &self.0 {
            HostMemory::Pinned(pinned) => pinned.slice(bytes),
            HostMemory::Pageable(memory) => memory[bytes].into(),
        }))
    }

    /// `bytes` of this memory, for a copy from a device to `.output(..)` into.
    pub fn slice_mut(&mut self, bytes: impl RangeBounds<usize>) -> Result<HostBufferSliceMut<'_>, Error> {
        let bytes =
            std::slice::try_range(bytes, ..self.len()).ok_or(Error::Invalid("the range is outside the host buffer"))?;
        Ok(HostBufferSliceMut(match &mut self.0 {
            HostMemory::Pinned(pinned) => pinned.slice_mut(bytes),
            HostMemory::Pageable(memory) => (&mut memory[bytes]).into(),
        }))
    }
}

impl Deref for HostBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.0 {
            HostMemory::Pinned(pinned) => pinned,
            HostMemory::Pageable(memory) => memory,
        }
    }
}

impl DerefMut for HostBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        match &mut self.0 {
            HostMemory::Pinned(pinned) => pinned,
            HostMemory::Pageable(memory) => memory,
        }
    }
}

impl Debug for HostBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostBuffer")
            .field("pinned", &self.is_pinned())
            .field("len", &self.len())
            .finish()
    }
}

/// A copy of `bytes` in memory [`HostBuffer::zero`] provides.
impl<T: AsRef<[u8]> + ?Sized> From<&T> for HostBuffer {
    fn from(bytes: &T) -> Self {
        let mut host = Self::zero(bytes.as_ref().len());
        host.copy_from_slice(bytes.as_ref());
        host
    }
}

/// The vector's memory as it is: ordinary memory.
impl From<Vec<u8>> for HostBuffer {
    fn from(bytes: Vec<u8>) -> Self {
        Self(HostMemory::Pageable(bytes))
    }
}

impl From<HostBuffer> for Vec<u8> {
    fn from(host: HostBuffer) -> Self {
        match host.0 {
            HostMemory::Pinned(pinned) => pinned.to_vec(),
            HostMemory::Pageable(bytes) => bytes,
        }
    }
}

/// A copy in the same kind of memory, so every copy transfers as the original does; pinned memory
/// the system no longer provides falls back as [`HostBuffer::zero`] does.
impl Clone for HostBuffer {
    fn clone(&self) -> Self {
        match &self.0 {
            HostMemory::Pinned(pinned) => Self::from(&**pinned),
            HostMemory::Pageable(bytes) => Self(HostMemory::Pageable(bytes.clone())),
        }
    }
}

/// Host bytes a transfer writes to a device: part of a [`HostBuffer`], or any other bytes.
#[derive(Debug)]
pub struct HostBufferSlice<'a>(pub(crate) furiosa_opt_rt::HostBufferSlice<'a>);

/// Host bytes a transfer reads from a device into: part of a [`HostBuffer`], or any other bytes.
#[derive(Debug)]
pub struct HostBufferSliceMut<'a>(pub(crate) furiosa_opt_rt::HostBufferSliceMut<'a>);

impl<'a, T: AsRef<[u8]> + ?Sized> From<&'a T> for HostBufferSlice<'a> {
    fn from(bytes: &'a T) -> Self {
        Self(bytes.as_ref().into())
    }
}

impl<'a, T: AsMut<[u8]> + ?Sized> From<&'a mut T> for HostBufferSliceMut<'a> {
    fn from(bytes: &'a mut T) -> Self {
        Self(bytes.as_mut().into())
    }
}

impl<'a> From<&'a HostBuffer> for HostBufferSlice<'a> {
    fn from(host: &'a HostBuffer) -> Self {
        host.slice(..).expect("every byte is inside the host buffer")
    }
}

impl<'a> From<&'a mut HostBuffer> for HostBufferSliceMut<'a> {
    fn from(host: &'a mut HostBuffer) -> Self {
        host.slice_mut(..).expect("every byte is inside the host buffer")
    }
}

impl<B: Backend> Debug for HbmBuffer<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A device's memory, per chip: how much it can allocate and how much is free now.
#[derive(Debug)]
pub struct Memory<'a, B: Runtime = CurrentBackend> {
    pub(super) device: &'a B::Device,
}

impl<B: Backend> Memory<'_, B> {
    /// Bytes per chip the device can allocate in all, live or free.
    pub fn capacity(&self) -> usize {
        B::capacity(self.device)
    }

    /// Free bytes per chip now; the snapshot reserves nothing for a later allocation.
    pub fn usage(&self) -> Usage {
        B::usage(self.device)
    }
}
