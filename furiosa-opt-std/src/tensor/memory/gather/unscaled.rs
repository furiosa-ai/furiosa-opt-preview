use super::*;

impl<D: MaterializableScalar, Chip: M, TableAxes: M, B: Backend> HbmTensor<D, Chip, TableAxes, B> {
    /// Gathers row positions from DM; native lowering places the index for the output's PEs.
    #[primitive(HbmTensor::dma_gather_unscaled)]
    pub fn dma_gather_unscaled<IdxCluster: M, IdxSlice: M, IdxElement: M, Cluster: M, Slice: M, Element: M>(
        &self,
        index: &DmTensor<i32, Chip, IdxCluster, IdxSlice, IdxElement, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B> {
        let domain = Mapping::from_terms(
            <m![{ IdxCluster }, { IdxSlice }, { IdxElement }]>::to_value()
                .axes()
                .into_iter()
                .map(AxisTerm::to_term),
        );
        let output = <m![{ Cluster }, { Slice }, { Element }]>::to_value();
        let payload = Mapping::from_terms(output.carve(&domain).axes().into_iter().map(AxisTerm::to_term));
        let indexed = Mapping::from_terms(
            TableAxes::to_value()
                .carve(&payload)
                .axes()
                .into_iter()
                .map(AxisTerm::to_term),
        );
        let payload = furiosa_opt_lower::gather_payload(&TableAxes::to_value(), &indexed)
            .unwrap_or_else(|error| panic!("{error}"));
        validate_placement::<Cluster, Slice, Element>(&payload, &domain);
        let mut output = DmTensor::from_parts(Tensor::zeroed());
        B::gather_table::<
            D,
            m![{ Chip }, { Cluster }, { Slice }, { Element }],
            m![{ Chip }, { IdxCluster }, { IdxSlice }, { IdxElement }],
        >(
            &B::load_memory::<D>(&self.memory()),
            &mut output.inner.inner,
            &index.inner.inner,
            &Pair::<Chip, TableAxes>::to_value(),
            &indexed,
            &Chip::to_value(),
            crate::backend::GatherRead {
                domain: &domain,
                prefix: None,
                unit: IndexUnit::Positions,
                batch_prefix: None,
            },
        );
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use crate::{Device, Topology};

    #[test]
    fn permutes_gathered_payload() {
        axes![Rows = 4, Left = 2, Right = 2, Key = 4];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let table =
            HbmTensor::<i32, m![1], m![Rows, Right, Left], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..16));
        let index = HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(&device.tdma, &Tensor::from_vec([2, 0, 2, 3]));
        let index: DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Key], Cpu> = index.to_dm(&mut device.tdma);

        let output: DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Left, Key, Right], Cpu> =
            table.dma_gather_unscaled(&index);

        assert_eq!(
            output
                .to_hbm::<m![Left, Key, Right]>(&mut device.tdma)
                .load()
                .into_vec(),
            [8, 10, 0, 2, 8, 10, 12, 14, 9, 11, 1, 3, 9, 11, 13, 15],
        );
    }

    #[test]
    fn preserves_repeated_positions() {
        axes![Rows = 4, Width = 4, Key = 4];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![Rows, Width], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..16));
        let index = HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(&device.tdma, &Tensor::from_vec([2, 0, 2, 3]));
        let index: DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Key], Cpu> = index.to_dm(&mut device.tdma);

        let output: DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Key, Width], Cpu> =
            table.dma_gather_unscaled(&index);

        assert_eq!(
            output.to_hbm::<m![Key, Width]>(&mut device.tdma).load().into_vec(),
            [8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
        );
    }

    /// Positions replicated on both clusters and over two slices each gather into every replica.
    #[test]
    fn fills_every_replica() {
        axes![Rows = 4, Width = 4, Key = 4];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![Rows, Width], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..16));
        let positions = [2, 0, 2, 3];
        let index = HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(&device.tdma, &Tensor::from_vec(positions));
        let index: DmTensor<i32, m![1], m![2], m![Key, 2 # 64], m![1 # 2], Cpu> = index.to_dm(&mut device.tdma);

        let output: DmTensor<i32, m![1], m![2], m![Key, 2 # 64], m![Width], Cpu> = table.dma_gather_unscaled(&index);

        let expected = (0..2).flat_map(|_cluster| {
            positions.into_iter().flat_map(|row| {
                (0..64).flat_map(move |replica| (0..4).map(move |w| if replica < 2 { row * 4 + w } else { 0 }))
            })
        });
        assert_eq!(output.inner.inner.to_vec(), expected.collect::<Vec<_>>());
    }
}
