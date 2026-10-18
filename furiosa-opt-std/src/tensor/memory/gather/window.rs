use super::*;

impl<D: MaterializableScalar, Chip: M, Element: M, B: Backend> slice::Slice<HbmTensor<D, Chip, Element, B>> {
    /// Gathers a field of each block while preserving block-relative offsets and entry selection.
    #[primitive(HbmTensor::gather_window)]
    pub fn gather_window<Axis: M, const LEN: usize, Window: M, Indexed: M, Payload: M>(
        &self,
        start: usize,
    ) -> GatherPlan<'_, D, Chip, Window, Indexed, Begin, B> {
        let rows = slice::outer::<Chip, Window>(2);
        let payload = validate_gather_payload(
            &rows.clone().pair(Window::to_value()),
            &rows.pair(Indexed::to_value()),
            &Payload::to_value(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let blocks = self
            .values
            .iter()
            .map(|block| block.view().tile::<Axis, LEN, Window>(start))
            .collect();
        GatherPlan {
            table: GatherTable {
                source: TableSource::Window(blocks),
                prefix: None,
            },
            payload,
            index: (),
            prefix: (),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    /// A field window preserves entry-relative byte offsets in an unpadded output.
    #[test]
    fn selects_field_without_padding() {
        axes![Token = 16, Field = 2, Width = 32, Key = 4];
        let mut context = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).expect("device");
        let blocks: Slice<HbmTensor<i32, m![1], m![Token, Field, Width], Cpu>> = (0..3)
            .map(|block| {
                HbmTensor::from_value(
                    &context.tdma,
                    &Tensor::from_vec((0..1024).map(move |at| block * 10000 + at)),
                )
            })
            .collect();
        let indices =
            HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(&context.tdma, &Tensor::from_vec([4096, 0, 4096, 0]));
        for field in 0..2 {
            let output: DmTensor<i32, m![1], m![Width / 16], m![Key, Token], m![Width % 16], Cpu> = blocks
                .gather_window::<m![Field], 1, m![Token, Field = 1 # 2, Width], m![1], m![Token, Field = 1 # 2, Width]>(
                    field,
                )
                .entry(1)
                .by_byte_offsets(&indices)
                .to_dm(&mut context.tdma);
            let field = i32::try_from(field).expect("field coordinate");
            let expected: Vec<_> = [2, 1, 2, 1]
                .into_iter()
                .flat_map(|block| {
                    (0..16).flat_map(move |token| {
                        (0..32).map(move |width| block * 10000 + token * 64 + field * 32 + width)
                    })
                })
                .collect();
            assert_eq!(
                output
                    .to_hbm::<m![Key, Token, Width]>(&mut context.tdma)
                    .load()
                    .into_vec(),
                expected
            );
        }
    }

    #[test]
    fn selects_field_after_entry() {
        axes![Token = 16, Field = 2, Width = 32, Key = 2];
        let mut context = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).expect("device");
        let blocks: Slice<HbmTensor<i32, m![1], m![Token, Field, Width], Cpu>> = (0..3)
            .map(|block| {
                HbmTensor::from_value(
                    &context.tdma,
                    &Tensor::from_vec((0..16).flat_map(move |token| {
                        (0..2).flat_map(move |field| {
                            (0..32).map(move |width| block * 10000 + token * 100 + field * 40 + width)
                        })
                    })),
                )
            })
            .collect();
        let indices = HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(&context.tdma, &Tensor::from_vec([4096, 0]));
        for field in 0..2 {
            let output: DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Key, Token, Width], Cpu> = blocks
                .gather_window::<m![Field], 1, m![Token, Field = 1 # 2, Width], m![1], m![Token, Field = 1 # 2, Width]>(
                    field,
                )
                .entry(1)
                .by_byte_offsets(&indices)
                .to_dm(&mut context.tdma);
            let field = i32::try_from(field).expect("field coordinate");
            let expected: Vec<_> = [2, 1]
                .into_iter()
                .flat_map(|block| {
                    (0..16).flat_map(move |token| {
                        (0..32).map(move |width| block * 10000 + token * 100 + field * 40 + width)
                    })
                })
                .collect();
            assert_eq!(
                output
                    .to_hbm::<m![Key, Token, Width]>(&mut context.tdma)
                    .load()
                    .into_vec(),
                expected
            );
        }
    }
}
