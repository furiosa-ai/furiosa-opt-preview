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

/// Source offsets for a transpose. An unchanged outer factor retains each physical copy while
/// the inner mapping changes. Carving the whole product could spend an anonymous outer broadcast
/// on newly added inner padding, then read the outer copies with stride zero.
pub(crate) fn transpose_config(src: &Mapping, dst: &Mapping, allow_broadcast: bool) -> SequencerConfig {
    let (src, dst) = &coordinates::narrow(src, dst);
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
        let mut config = transpose_config(src_inner, dst_inner, allow_broadcast);
        if src_outer.size() > 1 {
            config.0.insert(
                dst_inner.size(),
                SequencerEntry {
                    mapping: (**dst_outer).clone(),
                    memory_index: 0,
                    memory_stride: src_inner.size(),
                },
            );
        }
        return config;
    }
    let config = coordinates::sequence(&[src], &[dst], SequencerMode::Carve)
        .expect("transpose: Src must be covered by the dst stream")
        .remove(0);
    if !allow_broadcast {
        assert!(
            config
                .0
                .values()
                .all(|entry| entry.memory_stride != 0 || entry.mapping.is_padding()),
            "transpose: Dst adds a broadcast absent from Src",
        );
    }
    config
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

        let offsets = super::transpose_config(&source, &destination, true);

        assert_eq!(offsets.iter().collect::<Vec<_>>(), (0..128).collect::<Vec<_>>());
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

        let offsets = super::transpose_config(&source, &destination, false);

        assert_eq!(offsets.iter().collect::<Vec<_>>(), (0..128).collect::<Vec<_>>());
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

        let offsets = super::transpose_config(&source, &destination, false);

        assert_eq!(offsets.iter().collect::<Vec<_>>(), (0..128).collect::<Vec<_>>());
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
