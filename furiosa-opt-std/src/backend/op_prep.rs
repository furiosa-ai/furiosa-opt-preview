//! Per-op preparation step for the VALUE backends.
//!
//! Each helper derives the mapping data an op's host-side loop needs (transpose strides, scatter
//! payload, ...) and panics on a mapping its loop cannot walk.
//!
//! All helpers are noun-style and return a derived `Mapping` (or its parts). The shared file keeps
//! the value backends in lockstep on the mapping algebra without duplicating the prep step.
//!
//! [`Backend`]: crate::backend::Backend
//! [`Tensor`]: crate::tensor::Tensor

use std::collections::HashSet;

use furiosa_mapping::*;
use rayon::iter::Either;

pub(crate) mod coordinates;

pub(crate) fn broadcast_padding(mapping: Mapping) -> Mapping {
    use abi_stable::std_types::RBox;

    match mapping {
        Mapping::Padding {
            inner,
            padding,
            kind: PaddingKind::Top,
        } if matches!(&*inner, Mapping::Broadcast { size: 1 }) => Mapping::Broadcast { size: padding },
        Mapping::Pair { left, right } => {
            broadcast_padding(RBox::into_inner(left)).pair(broadcast_padding(RBox::into_inner(right)))
        }
        Mapping::Stride { inner, stride } => broadcast_padding(RBox::into_inner(inner)).stride(stride),
        Mapping::Modulo { inner, modulo } => broadcast_padding(RBox::into_inner(inner)).modulo(modulo),
        Mapping::Resize { inner, resize } => broadcast_padding(RBox::into_inner(inner)).resize(resize),
        Mapping::Padding { inner, padding, kind } => broadcast_padding(RBox::into_inner(inner)).padding(padding, kind),
        Mapping::Symbol { .. } | Mapping::Broadcast { .. } => mapping,
    }
}

/// Where each position of a transpose's destination reads its source.
pub(crate) enum Reads {
    /// The sequencer's walk over the source.
    Walk(SequencerConfig),
    /// The source position of each destination position, for a copy no single walk addresses, such
    /// as runs that straddle each other's rows. `None` marks a padding cell.
    Cells(Vec<Option<usize>>),
}

impl Reads {
    /// Each destination cell reads the source cell at its coordinates, and every source cell is read.
    fn cells(src: &Mapping, dst: &Mapping) -> Self {
        // Each symbol at its declared size, so that no destination coordinate wraps onto another cell.
        let mut axes: Vec<AxisTerm> = Vec::new();
        for axis in src.axes() {
            if axes.iter().all(|kept| kept.symbol != axis.symbol) {
                axes.push(AxisTerm {
                    stride: 1,
                    modulo: axis.size,
                    ..axis
                });
            }
        }
        let mut sources = vec![None; axes.iter().map(|axis| axis.size).product()];
        for (position, offset) in src.iter(&axes, &Index::new(), true).enumerate() {
            if let Some(offset) = offset {
                sources[offset].get_or_insert(position);
            }
        }
        let offsets: Vec<Option<usize>> = dst.iter(&axes, &Index::new(), true).collect();
        let mut read = vec![false; sources.len()];
        for &offset in offsets.iter().flatten() {
            read[offset] = true;
        }
        assert!(
            sources
                .iter()
                .zip(&read)
                .all(|(source, &read)| source.is_none() || read),
            "transpose: Src must be covered by the dst stream"
        );
        Self::Cells(
            offsets
                .into_iter()
                .map(|offset| {
                    offset.map(|offset| sources[offset].expect("transpose: Dst reads a cell absent from Src"))
                })
                .collect(),
        )
    }

    /// The number of destination positions read.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Walk(config) => config.stream_size(),
            Self::Cells(cells) => cells.len(),
        }
    }

    /// The source positions that destination positions `start..end` read.
    pub(crate) fn range(&self, start: usize, end: usize) -> impl Iterator<Item = usize> + '_ {
        match self {
            Self::Walk(config) => Either::Left(config.iter_range(start, end)),
            Self::Cells(cells) => Either::Right(cells[start..end].iter().map(|cell| cell.unwrap_or(0))),
        }
    }

    /// Whether one source position feeds two live destination positions: a walk entry that reads
    /// live cells at stride zero, or two cells that read one position.
    fn broadcasts(&self) -> bool {
        match self {
            Self::Walk(config) => config
                .0
                .values()
                .any(|entry| entry.memory_stride == 0 && !entry.mapping.is_padding()),
            Self::Cells(cells) => {
                let mut read = vec![false; cells.iter().flatten().max().map_or(0, |&last| last + 1)];
                cells
                    .iter()
                    .flatten()
                    .any(|&cell| std::mem::replace(&mut read[cell], true))
            }
        }
    }
}

/// The reads of a transpose: the sequencer's walk where one addresses the copy, else each destination
/// cell reads the source cell at its coordinates. Either way every source cell is read.
pub(crate) fn transpose_reads(src: &Mapping, dst: &Mapping, allow_broadcast: bool) -> Reads {
    let (src, dst) = &coordinates::narrow(src, dst);
    // An unchanged outer factor keeps each physical copy: carving the whole product could spend an
    // anonymous outer broadcast on newly added inner padding, then read its copies at stride zero.
    if let (
        Mapping::Pair {
            left: src_outer,
            right: src_inner,
        },
        Mapping::Pair {
            left: dst_outer,
            right: dst_inner,
        },
    ) = (src, dst)
        && src_outer == dst_outer
    {
        let copy_stride = src_inner.size();
        return match transpose_reads(src_inner, dst_inner, allow_broadcast) {
            Reads::Walk(mut config) => {
                if src_outer.size() > 1 {
                    config.0.insert(
                        dst_inner.size(),
                        SequencerEntry {
                            mapping: (**dst_outer).clone(),
                            memory_index: 0,
                            memory_stride: copy_stride,
                        },
                    );
                }
                Reads::Walk(config)
            }
            Reads::Cells(cells) => Reads::Cells(
                (0..src_outer.size())
                    .flat_map(|copy| cells.iter().map(move |cell| cell.map(|cell| copy * copy_stride + cell)))
                    .collect(),
            ),
        };
    }
    let reads = match coordinates::sequence(&[src], &[dst], SequencerMode::Carve) {
        Ok(mut configs) => Reads::Walk(configs.remove(0)),
        Err(_) => Reads::cells(src, dst),
    };
    assert!(
        allow_broadcast || !reads.broadcasts(),
        "transpose: Dst adds a broadcast absent from Src"
    );
    reads
}

/// The broadcast portion of `dst` for `reduce`: the axes whose symbol is absent in `src` (broadcast
/// axes are always new). Each [`AxisTerm`] is a single resolved symbol, so a dst axis is wholly new or
/// wholly shared, there is no straddling split (a composite spanning both) to guard against.
pub(crate) fn broadcast_axes(src: &Mapping, dst: &Mapping) -> Mapping {
    let src_idents: HashSet<Ident> = src.idents().into_iter().collect();
    let axes: Vec<Term> = dst
        .axes()
        .into_iter()
        .filter(|term| !src_idents.contains(&term.symbol))
        .map(AxisTerm::to_term)
        .collect();
    Mapping::from_terms(axes)
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::{broadcast_axes, broadcast_padding};
    use crate::backend::Cpu;
    use crate::tensor::Tensor;

    axes![A = 4, B = 2, C = 8];

    mod capacity {
        use furiosa_mapping::*;
        furiosa_mapping::axes![Rows = 512];
    }

    mod tile {
        use furiosa_mapping::*;
        furiosa_mapping::axes![Rows = 128];
    }

    mod storage {
        use furiosa_mapping::*;
        furiosa_mapping::axes![Rows = 1024];
    }

    /// A prefix copy writes each source cell once and preserves the capacity's remaining cells.
    #[test]
    fn copies_capacity_prefix() {
        use capacity::Rows as Source;
        use storage::Rows as Target;
        let source = Tensor::<i32, m![Source / 8, C, Source % 8], Cpu>::from_vec(0..4096);
        let mut target = Tensor::<i32, m![Target / 8, C, Target % 8], Cpu>::from_vec([-7; 8192]);
        target
            .view_mut()
            .tile::<m![Target / 8], m![Target / 8 = 64 #{!} 128, C, Target % 8], 64>(0)
            .transpose(source.view(), false);
        let output = target.into_vec();
        assert_eq!(&output[..4096], &(0..4096).collect::<Vec<_>>());
        assert_eq!(&output[4096..], &[-7; 4096]);
    }

    #[test]
    fn copies_loaded_rows() {
        use crate::prelude::*;
        use capacity::Rows as Capacity;
        use tile::Rows as Tile;

        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 4 }).unwrap();
        let source = HbmTensor::<i32, m![1], m![Capacity], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..512));

        let loaded: DmTensor<i32, m![1], m![1], m![Tile / 2], m![Tile % 2], Cpu> = source
            .view()
            .tile::<m![Capacity / Tile], 1, m![Capacity / Tile = 1 # (Capacity / Tile), Capacity % Tile]>(1)
            .to_dm(&mut device.tdma);

        assert_eq!(loaded.inner.into_vec(), (128..256).collect::<Vec<_>>());
    }

    #[test]
    fn copies_distinct_rows() {
        let source = Mapping::Symbol {
            symbol: Ident::R,
            size: 512,
        }
        .modulo(128);
        let destination = Mapping::Symbol {
            symbol: Ident::R,
            size: 128,
        };

        let reads = super::transpose_reads(&source, &destination, true);

        assert_eq!(
            reads.range(0, reads.len()).collect::<Vec<_>>(),
            (0..128).collect::<Vec<_>>()
        );
    }

    #[test]
    fn copies_rows_without_broadcast() {
        let source = Mapping::Symbol {
            symbol: Ident::R,
            size: 512,
        }
        .modulo(128);
        let destination = Mapping::Symbol {
            symbol: Ident::R,
            size: 128,
        };

        let reads = super::transpose_reads(&source, &destination, false);

        assert_eq!(
            reads.range(0, reads.len()).collect::<Vec<_>>(),
            (0..128).collect::<Vec<_>>()
        );
    }

    #[test]
    fn copies_split_rows() {
        let source = Mapping::Symbol {
            symbol: Ident::R,
            size: 128,
        };
        let destination = Mapping::Symbol {
            symbol: Ident::R,
            size: 512,
        }
        .modulo(128)
        .stride(4)
        .pair(
            Mapping::Symbol {
                symbol: Ident::R,
                size: 512,
            }
            .modulo(4),
        );

        let reads = super::transpose_reads(&source, &destination, false);

        assert_eq!(
            reads.range(0, reads.len()).collect::<Vec<_>>(),
            (0..128).collect::<Vec<_>>()
        );
    }

    /// The cells of `[N / run, R, N % run]` over `N = 80, R = 2` in wire order, each holding its
    /// coordinates as `2 n + r`.
    fn runs(run: i32) -> impl Iterator<Item = i32> {
        (0..80 / run).flat_map(move |s| (0..2).flat_map(move |r| (0..run).map(move |e| 2 * (run * s + e) + r)))
    }

    /// Runs of 20 and 16 cells that straddle each other's rows move every cell to its coordinates,
    /// though no single sequencer walk addresses the copy.
    #[test]
    fn relayouts_straddled_runs() {
        axes![N = 80, R = 2];
        let source = Tensor::<i32, m![N / 20, R, N % 20], Cpu>::from_vec(runs(20));
        let moved: Tensor<i32, m![N / 16, R, N % 16], Cpu> = source.transpose(false);
        assert_eq!(moved.into_vec(), runs(16).collect::<Vec<_>>());
    }

    /// Each copy of an unchanged outer factor keeps its own cells through runs no walk addresses.
    #[test]
    fn keeps_straddled_copies() {
        axes![N = 80, R = 2];
        let copies = |run| runs(run).chain(runs(run).map(|cell| cell + 1000));
        let source = Tensor::<i32, m![2, N / 20, R, N % 20], Cpu>::from_vec(copies(20));
        let moved: Tensor<i32, m![2, N / 16, R, N % 16], Cpu> = source.transpose(false);
        assert_eq!(moved.into_vec(), copies(16).collect::<Vec<_>>());
    }

    #[test]
    fn preserves_split_copies() {
        let source = Tensor::<i32, m![2, A / 2, A % 2], Cpu>::from_vec(1..=8);
        let padded: Tensor<i32, m![2, A / 2, A % 2 # 4], Cpu> = source.transpose(false);
        assert_eq!(
            padded.clone().into_vec(),
            [1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0],
        );
        let restored: Tensor<i32, m![2, A / 2, A % 2], Cpu> = padded.transpose(false);
        assert_eq!(restored.into_vec(), (1..=8).collect::<Vec<_>>());
    }

    #[test]
    fn preserves_copy_holes() {
        let source = Tensor::<i32, m![2, A / 2 # 4, A % 2], Cpu>::from_vec([
            1, 2, 3, 4, -99, -99, -99, -99, 5, 6, 7, 8, -99, -99, -99, -99,
        ]);
        let padded: Tensor<i32, m![2, A / 2 # 4, A % 2 # 4], Cpu> = source.transpose(false);
        assert_eq!(
            padded.into_vec(),
            [
                1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
        );
    }

    #[test]
    fn reads_renamed_tile() {
        let source = Tensor::<i32, m![C], Cpu>::from_vec(0..8);
        // SAFETY: the two four-element rows occupy the original eight elements in wire order.
        let view = unsafe { source.view().reshape::<m![B, A]>() };
        let actual = view.tile::<m![B], m![B = 1 # 2, A], 1>(1).read();
        assert_eq!(&actual.into_vec()[..4], &[4, 5, 6, 7]);
    }

    #[test]
    fn writes_renamed_tile() {
        let mut target = Tensor::<i32, m![C], Cpu>::from_vec([-7; 8]);
        let source = Tensor::<i32, m![A], Cpu>::from_vec(1..=4);
        // SAFETY: the two four-element rows occupy the original eight elements in wire order.
        let view = unsafe { target.view_mut().reshape::<m![B, A]>() };
        view.tile::<m![B], m![B = 1 #{!} 2, A], 1>(1)
            .transpose(source.view(), true);
        assert_eq!(target.into_vec(), [-7, -7, -7, -7, 1, 2, 3, 4]);
    }

    #[test]
    fn preserves_padding_constraints() {
        for mapping in [
            <m![A # 8]>::to_value(),
            Mapping::identity().padding(8, PaddingKind::Zero),
            Mapping::identity().padding(8, PaddingKind::Bottom),
        ] {
            assert_eq!(broadcast_padding(mapping.clone()), mapping);
        }
    }

    /// `broadcast_axes` is the symbol-level narrowing of `dst.carve(src)`: it keeps the SAME live axes
    /// but drops the positional padding carve adds. Where its invariant holds (each dst axis wholly new
    /// or wholly shared), its axes must equal carve's live axes.
    #[test]
    fn broadcast_axes_matches_carve() {
        let cases: [(Mapping, Mapping); 4] = [
            (<m![A, C]>::to_value(), <m![A, B, C]>::to_value()), // B is the new (broadcast) axis
            (<m![A]>::to_value(), <m![A, B]>::to_value()),
            (<m![B]>::to_value(), <m![A, B, C]>::to_value()), // two new axes (A, C)
            (<m![A, B, C]>::to_value(), <m![A, B, C]>::to_value()), // no broadcast
        ];
        let sorted_axes = |m: &Mapping| {
            let mut a = m.axes();
            a.sort();
            a
        };
        for (src, dst) in cases {
            assert_eq!(
                sorted_axes(&broadcast_axes(&src, &dst)),
                sorted_axes(&dst.carve(&src)),
                "broadcast_axes(src={src:?}, dst={dst:?})"
            );
        }
    }
}
