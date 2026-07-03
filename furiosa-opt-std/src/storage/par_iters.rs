//! rayon parallel iterators over the mapping/sequencer walks.
//!
//! Keeps rayon out of `furiosa-mapping-types` / `furiosa-mapping`: those provide the seekable,
//! double-ended walks ([`SequencerIter`] and [`MappingIter`], whose `range` carves a sub-walk); these
//! wrap them in rayon [`Producer`]s whose `split_at` seeks each half. So a config's / mapping's offsets
//! drive standard rayon combinators (`zip` / `fold` / `reduce`) at the call site, the split powered by
//! the seek.
//!
//! Each `par_iter()` is the parallel peer of a sequential `iter()`: [`SequencerConfig::iter`] and
//! `MappingExt::iter` ([`ParSequencerExt::par_iter`] / [`ParMappingExt::par_iter`]).
//! (A true `rayon::IntoParallelIterator` impl is blocked by the orphan rule — both the trait and these
//! types are foreign here — so these are inherent-style extension methods.)

use furiosa_mapping::{AxisTerm, Index, Mapping, MappingExt, MappingIter, SequencerConfig, SequencerIter};
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

// --- SequencerConfig buffer offsets ---

/// `.par_iter()`: the config's buffer offsets as an indexed parallel iterator (one `usize` per stream
/// position, in position order). The parallel peer of [`SequencerConfig::iter`].
pub(crate) trait ParSequencerExt {
    fn par_iter(&self) -> ParSequencerIter<'_>;
}

impl ParSequencerExt for SequencerConfig {
    fn par_iter(&self) -> ParSequencerIter<'_> {
        ParSequencerIter {
            config: self,
            len: self.stream_size(),
        }
    }
}

pub(crate) struct ParSequencerIter<'a> {
    config: &'a SequencerConfig,
    len: usize,
}

impl ParallelIterator for ParSequencerIter<'_> {
    type Item = usize;

    fn drive_unindexed<C: UnindexedConsumer<usize>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len)
    }
}

impl IndexedParallelIterator for ParSequencerIter<'_> {
    fn len(&self) -> usize {
        self.len
    }

    fn drive<C: Consumer<usize>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<usize>>(self, callback: CB) -> CB::Output {
        callback.callback(ParSequencerProducer {
            config: self.config,
            start: 0,
            end: self.len,
        })
    }
}

/// A contiguous stream-position range `[start, end)` of a config's offsets; splits by seeking.
struct ParSequencerProducer<'a> {
    config: &'a SequencerConfig,
    start: usize,
    end: usize,
}

impl Producer for ParSequencerProducer<'_> {
    type Item = usize;
    type IntoIter = SequencerIter;

    fn into_iter(self) -> SequencerIter {
        self.config.iter_range(self.start, self.end)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.start + index;
        (
            ParSequencerProducer {
                config: self.config,
                start: self.start,
                end: mid,
            },
            ParSequencerProducer {
                config: self.config,
                start: mid,
                end: self.end,
            },
        )
    }
}

// --- Mapping cells ---

/// `.par_iter(axes, base, padding)`: a mapping's cells as an indexed parallel iterator (one
/// `Option<usize>` per cell, in position order — `None` for a padding cell). The parallel peer of
/// `MappingExt::iter`.
pub(crate) trait ParMappingExt {
    fn par_iter(&self, axes: &[AxisTerm], base: &Index, padding: bool) -> ParMappingIter;
    /// The natural self-walk in parallel: this mapping's own cells from the origin, padding included.
    /// The parallel peer of [`MappingCells::iter_cells`].
    fn par_iter_cells(&self) -> ParMappingIter;
}

impl ParMappingExt for Mapping {
    fn par_iter(&self, axes: &[AxisTerm], base: &Index, padding: bool) -> ParMappingIter {
        let iter = self.iter(axes, base, padding);
        let len = iter.len();
        ParMappingIter { iter, len }
    }

    fn par_iter_cells(&self) -> ParMappingIter {
        self.par_iter(&self.axes(), &Index::new(), true)
    }
}

/// `m.iter_cells()`: the natural self-walk — a mapping's own cells from the origin, padding included.
/// Collapses the recurring `m.iter(&m.axes(), &Index::new(), true)` idiom to one call. Yields one
/// `Option<usize>` per cell, in position order (`None` for a padding cell).
pub(crate) trait MappingCells {
    fn iter_cells(&self) -> MappingIter;
}

impl MappingCells for Mapping {
    fn iter_cells(&self) -> MappingIter {
        self.iter(&self.axes(), &Index::new(), true)
    }
}

/// Owns the full [`MappingIter`]; its `range` carves the per-worker sub-walks.
pub(crate) struct ParMappingIter {
    iter: MappingIter,
    len: usize,
}

impl ParallelIterator for ParMappingIter {
    type Item = Option<usize>;

    fn drive_unindexed<C: UnindexedConsumer<Option<usize>>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len)
    }
}

impl IndexedParallelIterator for ParMappingIter {
    fn len(&self) -> usize {
        self.len
    }

    fn drive<C: Consumer<Option<usize>>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Option<usize>>>(self, callback: CB) -> CB::Output {
        callback.callback(ParMappingProducer {
            iter: self.iter,
            start: 0,
            end: self.len,
        })
    }
}

/// A contiguous cell range `[start, end)` of a mapping walk; splits by seeking. Owns the full walk
/// (its plan is `RArc`-shared, so the clone in `split_at` is a refcount bump).
struct ParMappingProducer {
    iter: MappingIter,
    start: usize,
    end: usize,
}

impl Producer for ParMappingProducer {
    type Item = Option<usize>;
    type IntoIter = MappingIter;

    fn into_iter(self) -> MappingIter {
        self.iter.range(self.start, self.end)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.start + index;
        (
            ParMappingProducer {
                iter: self.iter.clone(),
                start: self.start,
                end: mid,
            },
            ParMappingProducer {
                iter: self.iter,
                start: mid,
                end: self.end,
            },
        )
    }
}
