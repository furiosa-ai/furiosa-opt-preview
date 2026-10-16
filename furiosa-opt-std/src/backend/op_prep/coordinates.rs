//! Matches tile-local coordinates across differently sized symbols of the same axis.

use abi_stable::std_types::RBox;
use furiosa_mapping::*;
use std::collections::{HashMap, HashSet};

pub(crate) fn narrow(src: &Mapping, dst: &Mapping) -> (Mapping, Mapping) {
    fn symbols(mapping: &Mapping, out: &mut HashSet<(Ident, usize)>) {
        match mapping {
            Mapping::Symbol { symbol, size } => {
                out.insert((*symbol, *size));
            }
            Mapping::Pair { left, right } => {
                symbols(left, out);
                symbols(right, out);
            }
            Mapping::Stride { inner, .. }
            | Mapping::Modulo { inner, .. }
            | Mapping::Resize { inner, .. }
            | Mapping::Padding { inner, .. } => symbols(inner, out),
            Mapping::Broadcast { .. } => {}
        }
    }
    fn rename(mapping: Mapping, other: &HashSet<(Ident, usize)>) -> Mapping {
        match mapping {
            Mapping::Modulo { inner, modulo } => match RBox::into_inner(inner) {
                Mapping::Symbol { symbol, size }
                    if size > modulo && other.contains(&(symbol, modulo)) && !other.contains(&(symbol, size)) =>
                {
                    Mapping::Symbol { symbol, size: modulo }
                }
                inner => rename(inner, other).modulo(modulo),
            },
            Mapping::Pair { left, right } => {
                rename(RBox::into_inner(left), other).pair(rename(RBox::into_inner(right), other))
            }
            Mapping::Stride { inner, stride } => rename(RBox::into_inner(inner), other).stride(stride),
            Mapping::Resize { inner, resize } => rename(RBox::into_inner(inner), other).resize(resize),
            Mapping::Padding { inner, padding, kind } => rename(RBox::into_inner(inner), other).padding(padding, kind),
            Mapping::Symbol { .. } | Mapping::Broadcast { .. } => mapping,
        }
    }
    let (mut in_src, mut in_dst) = (HashSet::new(), HashSet::new());
    symbols(src, &mut in_src);
    symbols(dst, &mut in_dst);
    (rename(src.clone(), &in_dst), rename(dst.clone(), &in_src))
}

fn unify(src: &Mapping, dst: &Mapping) -> (Mapping, Mapping) {
    fn sizes(mapping: &Mapping, out: &mut HashMap<Ident, usize>) {
        match mapping {
            Mapping::Symbol { symbol, size } => {
                out.entry(*symbol)
                    .and_modify(|largest| *largest = (*largest).max(*size))
                    .or_insert(*size);
            }
            Mapping::Pair { left, right } => {
                sizes(left, out);
                sizes(right, out);
            }
            Mapping::Stride { inner, .. }
            | Mapping::Modulo { inner, .. }
            | Mapping::Resize { inner, .. }
            | Mapping::Padding { inner, .. } => sizes(inner, out),
            Mapping::Broadcast { .. } => {}
        }
    }
    fn rename(mapping: Mapping, largest: &HashMap<Ident, usize>) -> Mapping {
        match mapping {
            Mapping::Symbol { symbol, size } if largest[&symbol] > size => Mapping::Symbol {
                symbol,
                size: largest[&symbol],
            }
            .modulo(size),
            Mapping::Pair { left, right } => {
                rename(RBox::into_inner(left), largest).pair(rename(RBox::into_inner(right), largest))
            }
            Mapping::Stride { inner, stride } => rename(RBox::into_inner(inner), largest).stride(stride),
            Mapping::Modulo { inner, modulo } => rename(RBox::into_inner(inner), largest).modulo(modulo),
            Mapping::Resize { inner, resize } => rename(RBox::into_inner(inner), largest).resize(resize),
            Mapping::Padding { inner, padding, kind } => {
                rename(RBox::into_inner(inner), largest).padding(padding, kind)
            }
            Mapping::Symbol { .. } | Mapping::Broadcast { .. } => mapping,
        }
    }
    let mut largest = HashMap::new();
    sizes(src, &mut largest);
    sizes(dst, &mut largest);
    (rename(src.clone(), &largest), rename(dst.clone(), &largest))
}

pub(crate) fn sequence(
    src: &[&Mapping],
    dst: &[&Mapping],
    mode: SequencerMode,
) -> Result<Vec<SequencerConfig>, SequencerError> {
    let combined = |maps: &[&Mapping]| {
        maps.iter()
            .fold(Mapping::identity(), |combined, map| combined.pair((*map).clone()))
    };
    let (whole_src, whole_dst) = (combined(src), combined(dst));
    let src: Vec<_> = src.iter().map(|map| narrow(map, &whole_dst).0).collect();
    let dst: Vec<_> = dst.iter().map(|map| narrow(&whole_src, map).1).collect();
    furiosa_mapping::sequence(&src.iter().collect::<Vec<_>>(), &dst.iter().collect::<Vec<_>>(), mode).or_else(|_| {
        let (whole_src, whole_dst) = (
            combined(&src.iter().collect::<Vec<_>>()),
            combined(&dst.iter().collect::<Vec<_>>()),
        );
        let src: Vec<_> = src.iter().map(|map| unify(map, &whole_dst).0).collect();
        let dst: Vec<_> = dst.iter().map(|map| unify(&whole_src, map).1).collect();
        furiosa_mapping::sequence(&src.iter().collect::<Vec<_>>(), &dst.iter().collect::<Vec<_>>(), mode)
    })
}
