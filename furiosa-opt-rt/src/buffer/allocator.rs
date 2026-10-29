use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::{Allocation, Buffer, Owner};
use crate::arch::rngd;
use crate::driver::dram::PAGE;
use crate::driver::mailbox::IN_FLIGHT;
use crate::memory::{Kind, Usage};
use crate::{Error, Result, Topology};

/// One device's memory allocator, over its DRAM and SRAM. DRAM grows up, shared inside the peer aperture and
/// chip-local from its end, so back-to-back buffers ascend; SRAM residents grow down. A
/// [`Device`](crate::Device) allocates over the ranges its boot leaves free.
pub struct Allocator {
    topology: Topology,
    dram: Mutex<Spans>,
    sram: Mutex<Sram>,
    /// Where shared DRAM ends, rounded down to whole allocation units.
    shared_end: usize,
    /// Where device code addresses physical SRAM offset zero.
    sram_base: usize,
    /// How many ordered owners [`Allocator::owner`] has handed out.
    owners: AtomicU64,
    /// How many allocations this allocator made, which numbers the next.
    made: AtomicU64,
}

impl Allocator {
    /// Allocators for a host standing in for a device of `topology`, over fixed upper bounds of
    /// what such a device hands out: all the DRAM the driver offers its PEs, shared up to the
    /// peer windows' widest reach, and the whole SRAM window. A device hands out less, since its
    /// firmware, the firmware's DMA descriptors and its peer windows take some.
    pub fn new(topology: Topology) -> Self {
        // The driver splits a chip's DRAM evenly among its PEs, in whole pages.
        let per_pe = (rngd::DRAM as usize / usize::from(rngd::PES)) & !(PAGE - 1);
        let dram = 0..per_pe * usize::from(topology.pes);
        let shared_end = match topology.chips {
            0 | 1 => dram.end,
            // A chip's translation entries map its own DRAM first, then share the rest evenly
            // among its peers, so each peer window covers that many entries' worth.
            chips => {
                let chips = usize::from(chips);
                (rngd::ATU_ENTRIES - chips) / chips * rngd::ATU_ENTRY
            }
        };
        Self::over(topology, Ranges::new(dram, shared_end))
    }

    /// Hands out `ranges` of a device of `topology`.
    pub(crate) fn over(topology: Topology, ranges: Ranges) -> Self {
        Self {
            topology,
            shared_end: ranges.shared_end / DRAM_ALIGNMENT * DRAM_ALIGNMENT,
            dram: Mutex::new(Spans::new(ranges.dram, DRAM_ALIGNMENT)),
            sram_base: ranges.sram_base,
            sram: Mutex::new(Sram {
                spans: Spans::new(ranges.sram, ranges.sram_alignment),
                alignment: ranges.sram_alignment,
                reserved: vec![None; IN_FLIGHT].into(),
            }),
            owners: AtomicU64::new(0),
            made: AtomicU64::new(0),
        }
    }

    /// `requested` bytes of `kind` memory at one offset on every chip, free again once the
    /// buffer's last owner drops; [`Error::OutOfMemory`] when no free span holds them, and
    /// [`Error::SramOverlap`] when an SRAM resident would reach a launch's temporaries.
    pub fn alloc(self: &Arc<Self>, kind: Kind, requested: usize) -> Result<Buffer> {
        let full = Error::OutOfMemory { kind, requested };
        let at = match kind {
            Kind::Dram | Kind::DramShared => {
                let len = requested
                    .max(1)
                    .checked_next_multiple_of(DRAM_ALIGNMENT)
                    .ok_or(full.clone())?;
                let alignment = if len >= DRAM_UNIT { DRAM_UNIT } else { DRAM_ALIGNMENT };
                let mut allocator = self.dram();
                let at = if kind == Kind::DramShared {
                    allocator.gaps().find_map(|gap| {
                        let at = gap.start.checked_next_multiple_of(alignment)?;
                        (at.saturating_add(len) <= gap.end.min(self.shared_end)).then_some(at)
                    })
                } else {
                    // Chip-local memory leaves the aperture, the only DRAM shared memory can use, to
                    // shared allocations as far as the free spans allow.
                    allocator.above(self.shared_end, len, alignment)
                }
                .ok_or(full)?;
                debug_assert!(at.is_multiple_of(DRAM_ALIGNMENT), "peer DMA needs aligned DRAM");
                allocator.live.insert(at, len);
                at
            }
            Kind::Sram => {
                let mut sram = self.sram();
                let len = requested
                    .max(1)
                    .checked_next_multiple_of(sram.alignment)
                    .ok_or(full.clone())?;
                let at = sram.spans.top(len).ok_or(full)?;
                let temporaries = sram.temporaries();
                if at < temporaries {
                    return Err(Error::SramOverlap {
                        resident: at,
                        temporaries,
                    });
                }
                sram.spans.live.insert(at, len);
                at
            }
        };
        Ok(Buffer {
            offset: 0,
            len: requested,
            allocation: Arc::new(Allocation {
                at,
                generation: self.made.fetch_add(1, Ordering::Relaxed),
                kind,
                allocator: Arc::clone(self),
                active: Mutex::default(),
            }),
        })
    }

    /// Bytes per chip `kind` allocations can occupy in all, live or free.
    pub fn capacity(&self, kind: Kind) -> usize {
        match kind {
            Kind::Dram => self.dram().range.len(),
            Kind::DramShared => {
                let range = self.dram().range.clone();
                range.end.min(self.shared_end).saturating_sub(range.start)
            }
            Kind::Sram => self.sram().spans.range.len(),
        }
    }

    /// Free `kind` bytes per chip now: the free spans an allocation of `kind` can take.
    pub fn usage(&self, kind: Kind) -> Usage {
        let usage = |gaps: &mut dyn Iterator<Item = Range<usize>>| {
            gaps.fold(
                Usage {
                    available: 0,
                    largest: 0,
                },
                |usage, gap| Usage {
                    available: usage.available + gap.len(),
                    largest: usage.largest.max(gap.len()),
                },
            )
        };
        match kind {
            Kind::Dram => usage(&mut self.dram().gaps()),
            Kind::DramShared => usage(&mut self.dram().gaps().map(|gap| gap.start..gap.end.min(self.shared_end))),
            Kind::Sram => {
                let sram = self.sram();
                let floor = sram.temporaries().next_multiple_of(sram.alignment);
                usage(&mut sram.spans.gaps().map(|gap| gap.start.max(floor)..gap.end))
            }
        }
    }

    /// The chip count and processing elements per chip these allocations span.
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// Holds SRAM below `end` for a launch's temporaries until the reservation drops.
    pub(crate) fn reserve(self: &Arc<Self>, end: usize) -> Result<Reservation> {
        let mut sram = self.sram();
        let start = sram.spans.start();
        if end > start {
            return Err(Error::SramOverlap {
                resident: start,
                temporaries: end,
            });
        }
        // A reservation belongs to a launch holding an answer slot on every mailbox, and a mailbox has
        // `IN_FLIGHT` of them, the slot count the device passes.
        let slot = sram
            .reserved
            .iter()
            .position(Option::is_none)
            .expect("a reservation slot per admitted launch");
        sram.reserved[slot] = Some(end);
        Ok(Reservation {
            allocator: Arc::clone(self),
            slot,
        })
    }

    /// An ordered owner no earlier access of this device's memory had; its accesses order after
    /// every earlier owner's on the bytes they share.
    pub(crate) fn owner(&self) -> Result<Owner> {
        self.owners
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |owner| owner.checked_add(1))
            .map(Owner::Ordered)
            .map_err(|_| Error::Capacity)
    }

    pub(super) fn release(&self, kind: Kind, at: usize) {
        match kind {
            Kind::Sram => self.sram().spans.live.remove(&at),
            Kind::Dram | Kind::DramShared => self.dram().live.remove(&at),
        };
    }

    /// Where device code addresses physical SRAM offset zero.
    pub(super) fn sram_base(&self) -> usize {
        self.sram_base
    }

    // Every update under these locks is one map or slot write, so a lock a panic poisoned still
    // guards whole state.
    fn dram(&self) -> MutexGuard<'_, Spans> {
        self.dram.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn sram(&self) -> MutexGuard<'_, Sram> {
        self.sram.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl std::fmt::Debug for Allocator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Allocator")
            .field("topology", &self.topology)
            .finish_non_exhaustive()
    }
}

/// A temporary SRAM prefix, retained until its device work finishes.
pub(crate) struct Reservation {
    allocator: Arc<Allocator>,
    slot: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.allocator.sram().reserved[self.slot] = None;
    }
}

/// The device bytes one allocator hands out, as the driver found them.
#[derive(Clone)]
pub(crate) struct Ranges {
    /// DRAM offsets free on every chip.
    pub(crate) dram: Range<usize>,
    /// Where the DRAM every peer window covers ends.
    pub(crate) shared_end: usize,
    /// Physical SRAM offsets residents may occupy.
    pub(crate) sram: Range<usize>,
    /// Where device code addresses physical SRAM offset zero.
    pub(crate) sram_base: usize,
    pub(crate) sram_alignment: usize,
}

impl Ranges {
    pub(crate) fn new(dram: Range<usize>, shared_end: usize) -> Self {
        Self {
            dram,
            shared_end,
            sram: rngd::sram::WINDOW.start as usize..rngd::sram::WINDOW.end as usize,
            sram_base: (rngd::sram::Physical::START.virtual_addr() - rngd::sram::WINDOW.start) as usize,
            sram_alignment: rngd::sram::ALIGNMENT as usize,
        }
    }
}

struct Spans {
    range: Range<usize>,
    live: BTreeMap<usize, usize>,
}

impl Spans {
    /// Hands out `range`, narrowed to whole aligned units.
    fn new(range: Range<usize>, alignment: usize) -> Self {
        let start = range.start.next_multiple_of(alignment);
        let end = range.end / alignment * alignment;
        Self {
            range: start.min(end)..end,
            live: BTreeMap::new(),
        }
    }

    fn gaps(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        self.live
            .iter()
            .map(|(&at, &len)| at..at + len)
            .chain(std::iter::once(self.range.end..self.range.end))
            .scan(self.range.start, |end, occupied| {
                let gap = *end..occupied.start;
                *end = occupied.end;
                Some(gap)
            })
    }

    fn start(&self) -> usize {
        self.live.first_key_value().map_or(self.range.end, |(&at, _)| at)
    }

    fn top(&self, len: usize) -> Option<usize> {
        self.gaps()
            .filter(|gap| gap.len() >= len)
            .last()
            .map(|gap| gap.end - len)
    }

    /// The lowest `alignment`-aligned start of `len` free bytes among those that put the fewest of
    /// them below `floor`.
    fn above(&self, floor: usize, len: usize, alignment: usize) -> Option<usize> {
        self.gaps()
            .filter_map(|gap| {
                let low = gap.start.checked_next_multiple_of(alignment)?;
                let last = gap.end.checked_sub(len)? / alignment * alignment;
                (low <= last).then(|| {
                    let high = floor.next_multiple_of(alignment).clamp(low, last);
                    if high + len > floor { high } else { low }
                })
            })
            .min_by_key(|&at| (floor.saturating_sub(at).min(len), at))
    }
}

struct Sram {
    spans: Spans,
    alignment: usize,
    /// Where each admitted launch's temporaries end, one slot per launch the device admits at once.
    reserved: Box<[Option<usize>]>,
}

impl Sram {
    /// Where the temporaries of the admitted launches end: no resident may start below it.
    fn temporaries(&self) -> usize {
        self.reserved.iter().flatten().max().copied().unwrap_or(0)
    }
}

// PDMA P2P loses bandwidth when source and destination differ modulo 512.
const DRAM_ALIGNMENT: usize = 512;
// Kernels place their DRAM tensors on 4-KiB units of a buffer, so a buffer of a unit or more starts
// on one: a buffer that straddles the device's units costs a kernel time that depends on where
// earlier allocations left it.
const DRAM_UNIT: usize = 4096;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::rngd::sram;

    /// DRAM buffers of 4 KiB or more start on a 4-KiB unit, behind a smaller buffer too, while
    /// smaller buffers keep the 512-byte grain.
    #[test]
    fn aligns_large_dram_after_small_allocations() {
        for kind in [Kind::Dram, Kind::DramShared] {
            for requested in [4095_usize, 4096, 4097, 8192] {
                let allocator = crate::test::allocator(0..16384, 16384);
                let first = allocator.alloc(kind, 512).unwrap();
                let large = allocator.alloc(kind, requested).unwrap();
                let last = allocator.alloc(kind, 512).unwrap();
                let unit = if requested.next_multiple_of(DRAM_ALIGNMENT) >= DRAM_UNIT {
                    DRAM_UNIT
                } else {
                    DRAM_ALIGNMENT
                };
                assert!(
                    large.addr().is_multiple_of(unit),
                    "{kind:?} {requested}: {}",
                    large.addr()
                );
                assert!(last.addr().is_multiple_of(DRAM_ALIGNMENT));
                drop((first, large, last));
            }
        }
    }

    /// A peer aperture ending before free DRAM admits no shared allocation.
    #[test]
    fn refuses_empty_peer_apertures() {
        let allocator = crate::test::allocator(512..4096, 0);
        assert_eq!(allocator.capacity(Kind::DramShared), 0);
        assert_eq!(allocator.usage(Kind::DramShared).available, 0);
        assert!(matches!(
            allocator.alloc(Kind::DramShared, 1),
            Err(Error::OutOfMemory { .. })
        ));
    }

    /// Shared DRAM lies inside the peer aperture and draws on the same capacity as the rest; its
    /// capacity and usage report the aperture.
    #[test]
    fn shares_dram_capacity() {
        for end in [1024, 3072] {
            let allocator = crate::test::allocator(0..4096, end);
            let weights = allocator.alloc(Kind::Dram, 3072).unwrap();
            let shared = allocator.alloc(Kind::DramShared, 1024).unwrap();
            assert!(shared.addr() + shared.len() <= end);
            assert!(allocator.alloc(Kind::DramShared, 1).is_err());
            drop(weights);
            assert_eq!(allocator.alloc(Kind::DramShared, 1).is_ok(), end > 1024);
            drop(shared);
            assert!(allocator.alloc(Kind::DramShared, 1024).is_ok());
        }
        // Its capacity and free spans end at the aperture, as its allocations do.
        let allocator = crate::test::allocator(0..4096, 1024);
        let capacity = [Kind::Dram, Kind::DramShared].map(|kind| allocator.capacity(kind));
        assert_eq!(capacity, [4096, 1024]);
        let usage = allocator.usage(Kind::DramShared);
        assert_eq!((usage.available, usage.largest), (1024, 1024));
        assert!(allocator.alloc(Kind::DramShared, usage.largest + 1).is_err());
        let _shared = allocator.alloc(Kind::DramShared, usage.largest).unwrap();
        assert_eq!(allocator.usage(Kind::DramShared).largest, 0);
        assert_eq!(allocator.usage(Kind::Dram).available, 3072);
    }

    /// A device reports its capacity, and its free bytes and largest free span as they change.
    #[test]
    fn reports_free_spans() {
        let allocator = crate::test::allocator(1000..5121, 5121);
        let spans = || {
            let usage = allocator.usage(Kind::Dram);
            (usage.available, usage.largest)
        };
        assert_eq!(allocator.capacity(Kind::Dram), 4096);
        assert_eq!(spans(), (4096, 4096));
        let first = allocator.alloc(Kind::Dram, 1024).unwrap();
        let second = allocator.alloc(Kind::Dram, 1024).unwrap();
        drop(first);
        assert_eq!(spans(), (3072, 2048));
        assert!(allocator.alloc(Kind::Dram, 2049).is_err());
        let tail = allocator.alloc(Kind::Dram, 2048).unwrap();
        let head = allocator.alloc(Kind::Dram, 1024).unwrap();
        assert_eq!(spans(), (0, 0));
        drop((head, second, tail));
        assert_eq!(allocator.capacity(Kind::Dram), 4096);
        assert_eq!(spans(), (4096, 4096));
    }

    /// An oversized request is `OutOfMemory` with the length asked for, even at `usize::MAX`.
    #[test]
    fn rejects_oversized_requests() {
        let allocator = crate::test::allocator(0..4096, 4096);
        let _live = allocator.alloc(Kind::Sram, 1).unwrap();
        for requested in [sram::WINDOW.end as usize + 1, usize::MAX] {
            assert!(
                matches!(allocator.alloc(Kind::Sram, requested), Err(Error::OutOfMemory { requested: value, .. }) if value == requested)
            );
        }
        assert!(allocator.alloc(Kind::Sram, 1).is_ok());
    }

    /// A launch's SRAM temporaries never reach a resident, whichever is placed first.
    #[test]
    fn separates_launch_temporaries_from_residents() {
        let allocator = crate::test::allocator(0..4096, 4096);
        let end = sram::WINDOW.end as usize;
        let lower = allocator.reserve(end - 512).unwrap();
        let first = allocator.reserve(end).unwrap();
        let second = allocator.reserve(end).unwrap();
        drop(first);
        assert!(matches!(allocator.alloc(Kind::Sram, 1), Err(Error::SramOverlap { .. })));
        drop(second);
        assert_eq!(allocator.usage(Kind::Sram).largest, 512);
        let live = allocator.alloc(Kind::Sram, 512).unwrap();
        assert!(matches!(allocator.alloc(Kind::Sram, 1), Err(Error::SramOverlap { .. })));
        drop(lower);
        assert!(allocator.alloc(Kind::Sram, 1).is_ok());
        let start = live.addr() - sram::Physical::START.virtual_addr() as usize;
        assert!(allocator.reserve(start).is_ok());
        assert!(
            matches!(allocator.reserve(start + 1), Err(Error::SramOverlap { resident, temporaries }) if resident == start && temporaries == start + 1)
        );
    }
}
