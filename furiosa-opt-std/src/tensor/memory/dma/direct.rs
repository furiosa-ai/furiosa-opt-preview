//! Direct DMA between two placements.
//!
//! One copy with a known source and destination, as opposed to the indirect families next door
//! ([`super::gather`], [`super::scatter`]) and to [`crate::tensor::memory::redistribute`]. Every
//! conversion here checks the same thing before it moves anything: that the two placements lay their
//! elements out in an order one transfer can produce, which is [`super::layout`]'s rule.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_lower::DM_WRITE_ALIGN_BYTES;
use furiosa_opt_macro::primitive;

use super::layout::{assert_dm_dma_layout, assert_dma_layout};
use crate::Error;
use crate::backend::Backend;
use crate::constraints;
use crate::context::{Dma, DmaContext};
use crate::runtime::{Issued, Prepared, Submitted, Work};
use crate::scalar::{MaterializableScalar, Scalar};
use crate::tensor::Tensor;
use crate::tensor::memory::{
    DmTensor, DmTensorView, DmTensorViewMut, HbmTensor, HbmTensorView, HbmTensorViewMut, HostTensor, SpmTensor,
};
use std::future::Future;

impl<D: MaterializableScalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Prepares a transfer of this tensor to HBM: `.await` stages it into a fresh device
    /// allocation, `.output(&mut hbm).await` writes into an existing one.
    pub fn to_hbm<'t, Chip: M, Element2: M>(
        &'t self,
        dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    ) -> Prepared<ToHbm<'t, D, Element, Chip, Element2, B>> {
        Prepared::new(ToHbm {
            host: self,
            dma,
            _marker: PhantomData,
        })
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Prepares a transfer of this tensor to the host, which waits for the launches and transfers
    /// submitted before it that write the tensor: `.await` reads it into fresh host memory,
    /// `.output(&mut host).await` reads into an existing tensor, directly when that tensor is
    /// [`HostTensor::pinned`].
    pub fn to_host<'t, Element2: M>(
        &'t self,
        dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    ) -> Prepared<ToHost<'t, D, Chip, Element, Element2, B>>
    where
        D: MaterializableScalar,
    {
        Prepared::new(ToHost {
            hbm: self,
            dma,
            _marker: PhantomData,
        })
    }

    /// Converts to data memory tensor.
    #[primitive(HbmTensor::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        &self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        DmTensor::from_parts(self.read(dma).transpose(true))
    }
}

impl<D: MaterializableScalar, Chip: M, Element: M, B: Backend> HbmTensor<D, Chip, Element, B> {
    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(HbmTensor::to_hbm)]
    pub fn to_hbm<const DMA: Dma, Element2: M>(
        &self,
        dma: &mut DmaContext<{ DMA }, B>,
    ) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::from_value(dma, &self.read(dma).transpose(true))
    }
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend, Chips> HbmTensorView<'l, D, Chip, Element, B, Chips> {
    /// Writes to HBM tensor view. The destination's `Chip2` is free of the
    /// source's `Chip`: a read source (Top padding) may target a `view_mut`
    /// destination (Bottom padding). `transpose` validates the live layout.
    #[primitive(HbmTensorView::to_hbm_view)]
    pub fn to_hbm_view<const DMA: Dma, Chip2: M, Element2: M, Chips2>(
        self,
        dma: &mut DmaContext<{ DMA }, B>,
        dst: HbmTensorViewMut<'l, D, Chip2, Element2, B, Chips2>,
    ) {
        self.read(dma, |src| dst.write(dma, |mut dst| dst.transpose(src, true)));
    }

    /// Writes to data memory tensor view.
    #[primitive(HbmTensorView::to_dm_view)]
    pub fn to_dm_view<Chip2: M, Cluster: M, Slice: M, Element2: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster, Slice, Element2, B>,
    ) {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip2 }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        self.read(dma, |src| dst.inner.transpose(src, true));
    }
}

impl<'l, D: MaterializableScalar, Chip: M, Element: M, B: Backend, Chips>
    HbmTensorView<'l, D, Chip, Element, B, Chips>
{
    /// Converts to data memory tensor.
    #[primitive(HbmTensorView::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element2: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element2, B> {
        assert_dma_layout::<
            D,
            m![{ Chip }, { Element }],
            Element,
            m![{ Chip }, { Cluster }, { Slice }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        self.read(dma, |src| DmTensor::from_parts(src.read().transpose(true)))
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend> SpmTensor<D, Chip, Cluster, Pe, Element, B> {
    /// Converts to a fresh DM tensor.
    #[primitive(SpmTensor::to_dm)]
    pub fn to_dm<Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice2, Element2, B> {
        constraints::assert_pe_matches_slice::<Pe, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Pe }, { Element }],
            Element,
            m![{ Cluster }, { Slice2 }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        DmTensor::from_parts(self.inner.transpose(true))
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DmTensor<D, Chip, Cluster, Slice, Element, B> {
    /// Converts to an HBM tensor. The output region's address is assigned by the backend, not the
    /// caller.
    #[primitive(DmTensor::to_hbm)]
    pub fn to_hbm<Element2: M>(&self, dma: &mut DmaContext<{ Dma::Tensor }, B>) -> HbmTensor<D, Chip, Element2, B> {
        HbmTensor::from_value(dma, &self.inner.transpose(true))
    }

    /// Converts to a fresh SPM tensor with the caller-provided PE and element mappings.
    #[primitive(DmTensor::to_spm)]
    pub fn to_spm<Pe2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> SpmTensor<D, Chip, Cluster, Pe2, Element2, B> {
        assert_spm_staging::<D, Cluster, Slice, Element, Pe2, Element2>();
        SpmTensor::new(self.inner.transpose(true))
    }

    /// Converts to a data-memory tensor with the requested dimension mappings.
    #[primitive(DmTensor::to_dm)]
    pub fn to_dm<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        &self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip2, Cluster2, Slice2, Element2, B> {
        assert_dm_dma_layout::<D, Chip, Cluster, Slice, Element, Chip2, Cluster2, Slice2, Element2>(
            DM_WRITE_ALIGN_BYTES,
        );
        DmTensor::from_parts(self.inner.transpose(true))
    }
}

impl<'l, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend>
    DmTensorView<'l, D, Chip, Cluster, Slice, Element, B>
{
    /// Writes data to a mutable tensor view for HBM. `Chip2` is free of the
    /// source's `Chip`: a read source (Top) may target a `view_mut` destination
    /// (Bottom). `transpose` validates the live layout.
    #[primitive(DmTensorView::to_hbm_view)]
    pub fn to_hbm_view<Chip2: M, Element2: M, Chips>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        dst: HbmTensorViewMut<'l, D, Chip2, Element2, B, Chips>,
    ) {
        dst.write(dma, |mut dst| dst.transpose(self.inner, true));
    }

    /// Copies this DM view into a fresh SPM tensor with the requested PE and element mappings.
    #[primitive(DmTensorView::to_spm)]
    pub fn to_spm<Pe2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> SpmTensor<D, Chip, Cluster, Pe2, Element2, B> {
        assert_spm_staging::<D, Cluster, Slice, Element, Pe2, Element2>();
        let mut output = SpmTensor::new(Tensor::zeroed());
        output.inner.view_mut().transpose(self.inner, true);
        output
    }

    /// Writes data to a mutable tensor view for data memory. `Chip2`/`Cluster2`
    /// are free of the source's: a read source (Top) may target a `view_mut`
    /// destination (Bottom). `transpose` validates the live layout.
    #[primitive(DmTensorView::to_dm_view)]
    pub fn to_dm_view<Chip2: M, Cluster2: M, Slice2: M, Element2: M>(
        self,
        _dma: &mut DmaContext<{ Dma::Tensor }, B>,
        mut dst: DmTensorViewMut<'l, D, Chip2, Cluster2, Slice2, Element2, B>,
    ) {
        constraints::assert_dm_to_dm_dimension_preserved::<Chip, Chip2, Cluster, Cluster2, Slice, Slice2>();
        assert_dma_layout::<
            D,
            m![{ Cluster }, { Slice }, { Element }],
            Element,
            m![{ Cluster2 }, { Slice2 }, { Element2 }],
            Element2,
        >(DM_WRITE_ALIGN_BYTES);
        dst.inner.transpose(self.inner, true);
    }
}

/// The layout rule a DM source and an SPM destination must satisfy to stage in one transfer.
fn assert_spm_staging<D: Scalar, Cluster: M, Slice: M, Element: M, Pe2: M, Element2: M>() {
    constraints::assert_pe_matches_slice::<Pe2, Slice>();
    // The hardware config treats SPM as non-SRAM, so staging uses DRAM access width 1.
    assert_dma_layout::<
        D,
        m![{ Cluster }, { Slice }, { Element }],
        Element,
        m![{ Cluster }, { Pe2 }, { Element2 }],
        Element2,
    >(1);
}

/// A host-to-HBM transfer, as a [`Prepared`] holds it; see [`HostTensor::to_hbm`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ToHbm<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend> {
    host: &'t HostTensor<D, Element, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    _marker: PhantomData<(Chip, Element2)>,
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend>
    ToHbm<'t, D, Element, Chip, Element2, B>
{
    /// Writes the host tensor into `hbm`, relaid out on the host first unless it already lays the
    /// cells out as `hbm` does, since a backend moves bytes unchanged.
    async fn write(self, hbm: &mut HbmTensor<D, Chip, Element2, B>) -> Result<(), Error> {
        if same_layout::<Element, Pair<Chip, Element2>>() {
            return self
                .dma
                .to_hbm([(B::bytes(self.host.storage()), hbm.memory_mut())])
                .await;
        }
        let staged: Tensor<D, m![{ Chip }, { Element2 }], B> = self.host.inner().transpose(true);
        self.dma.to_hbm([(B::bytes(&staged.inner), hbm.memory_mut())]).await
    }
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend> Work
    for ToHbm<'t, D, Element, Chip, Element2, B>
{
    type Output = HbmTensor<D, Chip, Element2, B>;
    type Future = Issued<impl Future<Output = Result<Self::Output, Error>>>;

    fn submit(self) -> Submitted<Self::Future> {
        async move {
            let mut hbm = HbmTensor::alloc_in(self.dma.device(), crate::runtime::Kind::Dram)?;
            self.write(&mut hbm).await?;
            Ok(hbm)
        }
        .submit()
    }
}

impl<'t, D: MaterializableScalar, Element: M, Chip: M, Element2: M, B: Backend>
    Prepared<ToHbm<'t, D, Element, Chip, Element2, B>>
{
    /// Writes into `hbm`'s existing device allocation instead of a fresh one.
    pub fn output(
        self,
        hbm: &'t mut HbmTensor<D, Chip, Element2, B>,
    ) -> Prepared<impl Future<Output = Result<(), Error>>> {
        Prepared::new(self.into_work().write(hbm))
    }
}

/// An HBM-to-host transfer, as a [`Prepared`] holds it; see [`HbmTensor::to_host`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ToHost<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend> {
    hbm: &'t HbmTensor<D, Chip, Element, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    _marker: PhantomData<Element2>,
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend>
    ToHost<'t, D, Chip, Element, Element2, B>
{
    /// Reads the HBM tensor into `host`, relaid out on the host after unless `host` lays the cells
    /// out as the HBM tensor does, since a backend moves bytes unchanged.
    async fn read(self, host: &mut HostTensor<D, Element2, B>) -> Result<(), Error> {
        let from = self.hbm.memory();
        if same_layout::<Element2, Pair<Chip, Element>>() {
            return from.to_host(self.dma).output(B::bytes_mut(host.storage_mut())).await;
        }
        let mut staged = Tensor::<D, m![{ Chip }, { Element }], B>::zeroed();
        from.to_host(self.dma).output(B::bytes_mut(&mut staged.inner)).await?;
        *host = staged.transpose(true).into();
        Ok(())
    }
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend> Work
    for ToHost<'t, D, Chip, Element, Element2, B>
{
    type Output = HostTensor<D, Element2, B>;
    type Future = Issued<impl Future<Output = Result<Self::Output, Error>>>;

    fn submit(self) -> Submitted<Self::Future> {
        async move {
            let mut host = HostTensor::from(Tensor::zeroed());
            self.read(&mut host).await?;
            Ok(host)
        }
        .submit()
    }
}

impl<'t, D: MaterializableScalar, Chip: M, Element: M, Element2: M, B: Backend>
    Prepared<ToHost<'t, D, Chip, Element, Element2, B>>
{
    /// Reads into `host`'s own memory instead of fresh memory.
    pub fn output(self, host: &'t mut HostTensor<D, Element2, B>) -> Prepared<impl Future<Output = Result<(), Error>>> {
        Prepared::new(self.into_work().read(host))
    }
}

/// Whether `A` and `B` lay the same cells out in the same order, so their bytes agree.
fn same_layout<A: M, B: M>() -> bool {
    A::SIZE == B::SIZE && A::to_value().normalize() == B::to_value().normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::Cpu;
    use crate::runtime::{Device, Topology};

    #[test]
    fn cpu_to_dm_moves_a_chip_axis_into_the_element() {
        axes![A = 2, C = 2];

        let input = DmTensor::<i32, m![A], m![1], m![1 # 64], m![C], Cpu>::from_parts(Tensor::from_vec(
            (0..A::SIZE * 64 * C::SIZE).map(|value| value as i32),
        ));
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let output = input.to_dm::<m![2], m![1], m![1 # 64], m![A, C]>(&mut device.tdma);

        let output_chip_size = 64 * A::SIZE * C::SIZE;
        let mut expected = vec![0; 2 * output_chip_size];
        expected[..4].copy_from_slice(&[0, 1, 128, 129]);
        expected[output_chip_size..output_chip_size + 4].copy_from_slice(&[0, 1, 128, 129]);
        assert_eq!(output.view().inner.read().into_vec(), expected);
    }
}
