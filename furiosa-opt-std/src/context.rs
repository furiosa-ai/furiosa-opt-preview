use std::fmt::Debug;
use std::marker::ConstParamTy;
use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use super::scalar::{MaterializableScalar, Scalar};
use super::tensor::Tensor;
use super::tensor::memory::DmTensorView;
use super::tensor::tu::BeginTensor;
use crate::backend::{Backend, Runtime};
use crate::runtime::{HbmBufferSlice, HbmBufferSliceMut, HostBuffer};

/// Tensor units.
#[derive(Debug, PartialEq, Eq, ConstParamTy)]
pub enum Tu {
    /// Main context.
    Main,
    /// Sub context.
    Sub,
}

/// DMA units.
#[derive(Debug, PartialEq, Eq, ConstParamTy)]
pub enum Dma {
    /// Tensor DMA.
    Tensor,
    /// PCIe DMA.
    Pcie,
}

/// Context for a specific tensor unit.
#[primitive(TuContext)]
#[derive(Debug)]
pub struct TuContext<const T: Tu> {
    _marker: PhantomData<()>,
}

/// Context for a DMA engine.
#[primitive(DmaContext)]
pub struct DmaContext<const DMA: Dma, B: Runtime = crate::runtime::CurrentBackend> {
    device: B::Device,
    _marker: PhantomData<()>,
}

impl<const DMA: Dma, B: Runtime> Debug for DmaContext<DMA, B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DmaContext").finish_non_exhaustive()
    }
}

impl<const DMA: Dma, B: Runtime> DmaContext<DMA, B> {
    pub(crate) fn on(device: B::Device) -> Self {
        Self {
            device,
            _marker: PhantomData,
        }
    }

    pub(crate) fn device(&self) -> &B::Device {
        &self.device
    }
}

impl<const DMA: Dma, B: Backend> DmaContext<DMA, B> {
    pub(crate) fn load<D: Scalar>(&self, memory: HbmBufferSlice<'_, B>) -> B::Storage<D> {
        let mut host = HostBuffer::from_vec(vec![0; memory.region.host_len()]);
        now(B::read(&self.device, memory, (&mut host).into()))
            .expect("device-body DMA reads admitted memory into an equally sized host buffer");
        host.into()
    }

    pub(crate) fn store<D: Scalar>(&self, memory: HbmBufferSliceMut<'_, B>, value: &B::Storage<D>) {
        let host: HostBuffer = value.clone().into();
        now(B::write(&self.device, vec![((&host).into(), memory)]))
            .expect("device-body DMA writes an equally sized value into admitted memory");
    }
}

/// The output of a device-body transfer, which finishes on its first poll: only the CPU backend
/// runs a device body, and its transfers wait for nothing. Entering no executor, it runs under any.
fn now<T>(transfer: impl Future<Output = T>) -> T {
    match std::pin::pin!(transfer).poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
        std::task::Poll::Ready(output) => output,
        std::task::Poll::Pending => unreachable!("a device-body transfer finishes on its first poll"),
    }
}

impl<const T: Tu> TuContext<{ T }> {
    pub(crate) const fn on() -> Self {
        Self { _marker: PhantomData }
    }

    /// Begin a tensor unit operation in this context.
    #[primitive(TuContext::begin)]
    pub fn begin<'l, D: MaterializableScalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        &'l mut self,
        tensor: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
    ) -> BeginTensor<'l, { T }, D, Chip, Cluster, Slice, Identity, Element> {
        // The mappings differ only by `Identity`; `transmute` is a safe rewrap after the storage
        // mapping erasure.
        BeginTensor::new(self, tensor.inner.read().transmute())
    }

    /// Begin a tensor unit operation in this context with interleaved tensors.
    #[primitive(TuContext::begin_interleaved)]
    pub fn begin_interleaved<'l, I: AxisName, D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        &'l mut self,
        lhs: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
        rhs: DmTensorView<'l, D, Chip, Cluster, Slice, Element>,
    ) -> BeginTensor<'l, { T }, D, Chip, Cluster, Slice, Symbol<I>, Element> {
        let mut output = Tensor::<D, m![{ Chip }, { Cluster }, { Slice }, { Symbol<I> }, { Element }]>::zeroed();

        for (i, input) in [lhs, rhs].into_iter().enumerate() {
            output
                .view_mut()
                .tile_derived::<Symbol<I>, 1>(i)
                .transpose(input.inner, false);
        }

        BeginTensor::new(self, output)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{Cpu, internal::Backend};
    use crate::runtime::{Device, HbmBuffer, HostBuffer, Topology};
    use crate::scalar::i4;

    /// Device-body DMA preserves packed bytes across chips and aliased subranges.
    #[test]
    fn transfers_packed_aliases() {
        let device = Device::<Cpu>::new(Topology { chips: 2, pes: 8 }).unwrap();
        let buffer = HbmBuffer::alloc(&device, 4).unwrap();
        let mut alias = buffer.alias(1..3).unwrap();
        let bytes = vec![0x18, 0x7f, 0x20, 0x96];
        let value = <<Cpu as Backend>::Storage<i4>>::from(HostBuffer::from_vec(bytes.clone()));

        device.tdma.store(alias.slice_mut(.., ..).unwrap(), &value);
        let loaded = device.tdma.load::<i4>(buffer.slice(.., 1..3).unwrap());

        assert_eq!(&*HostBuffer::from(loaded), bytes);
    }
}
