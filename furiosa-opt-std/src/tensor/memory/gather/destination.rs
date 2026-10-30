use super::*;

impl<'l, D: MaterializableScalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, Domain: M, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedByteOffsets<IdxChip, Domain>, B>
{
    /// Gathers into the selected destination region, preserving its unwritten backing storage.
    #[primitive(GatherPlan::to_dm_view)]
    pub fn to_dm_view<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut destination: DmTensorViewMut<'_, D, Chip, Cluster, Slice, Element, B>,
    ) {
        let output: DmTensor<D, Chip, Cluster, Slice, Element, B> = self.materialize(dma, None);
        destination.inner.transpose(output.inner.view(), true);
    }
}

impl<'l, D: MaterializableScalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, Domain: M, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Prefixed<IndexedByteOffsets<IdxChip, Domain>>, B>
{
    /// Gathers the runtime prefix into the selected destination region, preserving the rest of its
    /// backing storage; entries of the region past the prefix are unspecified, as for `to_dm`.
    #[primitive(GatherPlan::to_dm_view)]
    pub fn to_dm_view<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut destination: DmTensorViewMut<'_, D, Chip, Cluster, Slice, Element, B>,
    ) {
        let output: DmTensor<D, Chip, Cluster, Slice, Element, B> = self.to_dm(dma);
        destination.inner.transpose(output.inner.view(), true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use crate::{Device, Topology};

    /// Writing either half preserves the other half and uses the requested table records.
    #[test]
    fn preserves_unselected_half() {
        axes![Rows = 4, Keys = 4, Head = 128];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 4 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![Rows, Head], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..512));
        let indices =
            HbmTensor::<i32, m![1], m![Keys % 2], Cpu>::from_value(&device.tdma, &Tensor::from_vec([1536, 512]));
        let mut output = DmTensor::<i32, m![1], m![1 # 2], m![1 # 64], m![Keys, Head], Cpu>::from_parts(
            Tensor::from_vec(std::iter::repeat_n(-17, 2 * 64 * 4 * 128)),
        );

        for half in 0..2 {
            table
                .gather::<m![Rows], m![Head]>()
                .by_byte_offsets(&indices)
                .to_dm_view(
                    &mut device.tdma,
                    output
                        .view_mut()
                        .tile::<m![Keys / 2], 1, m![Keys / 2 = 1 #{!} 2, Keys % 2, Head]>(half),
                );
            let actual = output.to_hbm::<m![Keys, Head]>(&mut device.tdma).load().into_vec();
            let expected: Vec<_> = (0..4)
                .flat_map(|key| {
                    (0..128).map(move |lane| {
                        if key / 2 <= half {
                            (if key % 2 == 0 { 384 } else { 128 }) + lane
                        } else {
                            -17
                        }
                    })
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    /// A prefixed gather writes its prefix into the selected half and preserves the other half.
    #[test]
    fn prefixed_preserves_unselected_half() {
        axes![Rows = 4, Keys = 4, Head = 128];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 4 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![Rows, Head], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..512));
        let indices =
            HbmTensor::<i32, m![1], m![Keys % 2], Cpu>::from_value(&device.tdma, &Tensor::from_vec([1536, 512]));
        let mut output = DmTensor::<i32, m![1], m![1 # 2], m![1 # 64], m![Keys, Head], Cpu>::from_parts(
            Tensor::from_vec(std::iter::repeat_n(-17, 2 * 64 * 4 * 128)),
        );

        table
            .gather::<m![Rows], m![Head]>()
            .by_byte_offsets(&indices)
            .sparse_prefix(1)
            .to_dm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![Keys / 2], 1, m![Keys / 2 = 1 #{!} 2, Keys % 2, Head]>(1),
            );
        let actual = output.to_hbm::<m![Keys, Head]>(&mut device.tdma).load().into_vec();
        assert!(actual[..2 * 128].iter().all(|&value| value == -17));
        assert_eq!(actual[2 * 128..3 * 128], (384..512).collect::<Vec<_>>()[..]);
    }
}
