use abi_stable::std_types::RBox;
use furiosa_mapping::*;

#[test]
fn finalization_combines_equivalent_composite_representations() {
    let a = Mapping::Symbol {
        symbol: Ident::A,
        size: 2,
    };
    let b = Mapping::Symbol {
        symbol: Ident::B,
        size: 3,
    };
    let base = a.pair(b);

    for kind in [PaddingKind::Top, PaddingKind::Zero, PaddingKind::Bottom] {
        let whole = base.clone().padding(16, kind);
        let baked = base.clone().stride(2).padding(8, kind);
        for (upper_mapping, upper_stride) in [(whole.clone(), 2), (baked, 1)] {
            for position in [3, 5, 6] {
                let mut index = Index::default();
                index.add_term(
                    Term {
                        inner: Atom::Composite(RBox::new(whole.clone())),
                        stride: 1,
                        modulo: 2,
                    },
                    position % 2,
                );
                index.add_term(
                    Term {
                        inner: Atom::Composite(RBox::new(upper_mapping.clone())),
                        stride: upper_stride,
                        modulo: 8,
                    },
                    position / 2,
                );
                let expected: Result<RSortedMap<Ident, usize>, PaddingKind> = match position {
                    3 => Ok([(Ident::A, 1)].into_iter().collect()),
                    5 => Ok([(Ident::A, 1), (Ident::B, 2)].into_iter().collect()),
                    6 => Err(kind),
                    _ => unreachable!(),
                };
                assert_eq!(
                    index.finalize().into_result(),
                    expected,
                    "position={position}, kind={kind:?}, upper_stride={upper_stride}",
                );
            }
        }
    }
}

#[test]
fn striding_a_partial_row_preserves_padding() {
    let a = Mapping::Symbol {
        symbol: Ident::A,
        size: 2,
    };

    for size in [5, 6] {
        let b = Mapping::Symbol { symbol: Ident::B, size };
        for kind in [PaddingKind::Top, PaddingKind::Zero, PaddingKind::Bottom] {
            let strided = a.clone().pair(b.clone().padding(8, kind)).resize(12).stride(2);
            assert_eq!(strided.index(3).finalize(PaddingKind::Bottom).into_result(), Err(kind));
            assert_eq!(
                strided.index(4).finalize(PaddingKind::Bottom).into_result(),
                Ok([(Ident::A, 1)].into_iter().collect()),
            );
        }
    }
}

#[test]
fn padding_queries_ignore_cells_outside_the_read() {
    let axis = Mapping::Symbol {
        symbol: Ident::A,
        size: 6,
    };
    for kind in [PaddingKind::Top, PaddingKind::Zero, PaddingKind::Bottom] {
        let padded = axis.clone().padding(8, kind);
        for read in [padded.clone().resize(6), padded.clone().modulo(4)] {
            assert!(read.has_no_padding(), "read={read}");
            assert!(!read.has_bottom_pad(), "read={read}");
        }
        let read = padded.resize(7);
        assert!(!read.has_no_padding(), "read={read}");
        assert_eq!(read.has_bottom_pad(), kind == PaddingKind::Bottom, "read={read}");
    }
}

#[test]
fn padding_queries_respect_outer_padding_in_strided_pairs() {
    let outer = Mapping::identity().padding(2, PaddingKind::Top);
    let inner = Mapping::identity().padding(2, PaddingKind::Bottom);
    let read = outer.pair(inner).stride(2);
    assert!(!read.has_bottom_pad());
    assert!(!read.has_no_padding());

    let outer = Mapping::Broadcast { size: 4 }.padding(6, PaddingKind::Bottom);
    let inner = Mapping::Broadcast { size: 4 }.padding(6, PaddingKind::Zero);
    let mapping = outer.pair(inner);
    for (extent, has_bottom) in [(24, false), (36, true)] {
        let read = mapping.clone().resize(extent).stride(4);
        assert!(!read.has_no_padding(), "read={read}");
        assert_eq!(read.has_bottom_pad(), has_bottom, "read={read}");
    }
}

#[test]
fn unittest_padding_queries_resolve_recombined_composites() {
    let a = Mapping::Symbol {
        symbol: Ident::A,
        size: 2,
    };
    let b = Mapping::Symbol {
        symbol: Ident::B,
        size: 2,
    };
    let c = Mapping::Symbol {
        symbol: Ident::C,
        size: 3,
    };
    for kind in [PaddingKind::Top, PaddingKind::Zero, PaddingKind::Bottom] {
        let x = b.clone().pair(c.clone().padding(4, kind));
        let read = x.clone().stride(2).pair(a.clone()).pair(x.modulo(2));
        for read in [read.clone(), read.normalize()] {
            assert_eq!(read.index(5).finalize(PaddingKind::Bottom).into_result(), Err(kind));
            assert!(!read.has_no_padding(), "read={read}");
            assert_eq!(read.has_bottom_pad(), kind == PaddingKind::Bottom, "read={read}");
            assert!(read.resize(4).has_no_padding());
        }
    }
    let digit = Mapping::Broadcast { size: 3 }
        .padding(4, PaddingKind::Zero)
        .padding(8, PaddingKind::Zero)
        .modulo(2);
    let read = digit.clone().pair(digit);
    assert_eq!(
        read.index(3).finalize(PaddingKind::Bottom).into_result(),
        Err(PaddingKind::Bottom)
    );
    assert!(read.has_bottom_pad());
    assert!(!read.has_no_padding());
}

#[test]
fn unittest_padding_queries_do_not_enumerate_broadcast_repetitions() {
    for kind in [PaddingKind::Top, PaddingKind::Zero, PaddingKind::Bottom] {
        let read = Mapping::Broadcast { size: 1 << 30 }
            .pair(Mapping::identity().padding(6, kind))
            .stride(4);
        assert!(!read.has_no_padding());
        assert_eq!(read.has_bottom_pad(), kind == PaddingKind::Bottom);
    }
    // Every sampled column is live. Proving absence must also skip the repeated rows.
    let read = Mapping::Broadcast { size: 1 << 30 }
        .pair(
            Mapping::Symbol {
                symbol: Ident::C,
                size: 3,
            }
            .padding(4, PaddingKind::Bottom),
        )
        .stride(2);
    assert!(read.has_no_padding());
    assert!(!read.has_bottom_pad());
}
