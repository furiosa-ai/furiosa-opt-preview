use std::future::Future;
use std::marker::PhantomData;

use furiosa_opt_macro::primitive;

use crate::Error;
use crate::backend::{Backend, Cpu};
use crate::context::{Dma, DmaContext};
use crate::runtime::{
    CurrentBackend, HbmBufferSlice, HbmBufferSliceMut, HostBuffer, HostBufferSlice, Prepared, Returned, Submitted, Work,
};
use crate::scalar::{RuntimeScalar, Scalar};

/// One scalar value replicated in the local HBM of every chip in the launch group.
#[primitive(HbmScalar)]
pub struct HbmScalar<D: RuntimeScalar, B: Backend = CurrentBackend> {
    /// The memory the value lives in, the same bytes on every chip.
    buffer: B::Buffer,
    value: PhantomData<D>,
}

impl<D: RuntimeScalar, B: Backend> std::fmt::Debug for HbmScalar<D, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HbmScalar").field("buffer", &self.buffer).finish()
    }
}

impl<D: RuntimeScalar, B: Backend> HbmScalar<D, B> {
    /// Prepares a copy of one host value to HBM: `.await` copies it into a fresh scalar,
    /// `.output(&mut scalar)` into an existing one.
    pub fn from_host(value: D, dma: &mut DmaContext<{ Dma::Pcie }, B>) -> Prepared<FromHost<'_, D, B>> {
        Prepared::new(FromHost {
            value,
            dma,
            to: Returned,
        })
    }

    /// Loads the value into an SPM scalar every chip and cluster can read.
    #[primitive(HbmScalar::to_spm)]
    pub fn to_spm(&self, dma: &mut DmaContext<{ Dma::Tensor }, B>) -> D {
        let bytes: HostBuffer = dma.load::<D::Storage>(self.memory()).into();
        D::from_storage(D::Storage::load(&bytes, 0))
    }
}

/// The host backend's device memory is host memory, so its scalars read synchronously.
impl<D: RuntimeScalar> HbmScalar<D, Cpu> {
    /// The scalar's packed HBM image on one chip.
    pub fn to_buf(&self) -> Vec<u8> {
        let memory = self.memory();
        let bytes = memory.buffer.load(&memory.region);
        Self::encode(D::from_storage(D::Storage::load(&bytes, 0)))
    }
}

impl<D: RuntimeScalar, B: Backend> HbmScalar<D, B> {
    /// A scalar in a new, uninitialized allocation on `device`.
    pub(crate) fn alloc_in(device: &B::Device) -> Result<Self, Error> {
        Ok(Self {
            buffer: B::alloc(device, crate::runtime::Kind::Dram, D::Storage::buf_bytes(1))?,
            value: PhantomData,
        })
    }

    /// The memory the value lives in.
    pub(crate) fn memory(&self) -> HbmBufferSlice<'_, B> {
        HbmBufferSlice::whole(&self.buffer)
    }

    pub(crate) fn memory_mut(&mut self) -> HbmBufferSliceMut<'_, B> {
        HbmBufferSliceMut::whole(&mut self.buffer)
    }

    pub(crate) fn encode(value: D) -> Vec<u8> {
        D::Storage::to_buf(&[value.into_storage()])
    }

    /// Submits a copy of `value` into this scalar on every chip.
    fn copy_from(
        &mut self,
        device: &B::Device,
        value: D,
    ) -> impl Future<Output = Result<(), Error>> + Send + 'static + use<D, B> {
        let memory = self.memory_mut();
        let bytes = Self::encode(value).repeat(memory.region.chips.count_ones() as usize);
        B::write(device, vec![(HostBufferSlice::from(&bytes), memory)])
    }
}

/// A host value's copy to HBM, as a [`Prepared`] holds it: into a fresh scalar, or into `To` once
/// [`Prepared::output`] names one; see [`HbmScalar::from_host`].
#[doc(hidden)]
#[derive(Debug)]
pub struct FromHost<'t, D: RuntimeScalar, B: Backend, To = Returned> {
    value: D,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
    to: To,
}

impl<D: RuntimeScalar, B: Backend> Work for FromHost<'_, D, B> {
    type Output = HbmScalar<D, B>;
    type Future = impl Future<Output = Result<HbmScalar<D, B>, Error>> + 'static;

    fn submit(self) -> Submitted<Self::Future> {
        let device = self.dma.device();
        let written =
            HbmScalar::<D, B>::alloc_in(device).map(|mut scalar| (scalar.copy_from(device, self.value), scalar.buffer));
        Submitted(async move {
            let (written, buffer) = written?;
            written.await?;
            Ok(HbmScalar {
                buffer,
                value: PhantomData,
            })
        })
    }
}

impl<D: RuntimeScalar, B: Backend> Work for FromHost<'_, D, B, &mut HbmScalar<D, B>> {
    type Output = ();
    type Future = impl Future<Output = Result<(), Error>> + Send + 'static;

    fn submit(self) -> Submitted<Self::Future> {
        Submitted(self.to.copy_from(self.dma.device(), self.value))
    }
}

impl<'t, D: RuntimeScalar, B: Backend> Prepared<FromHost<'t, D, B>> {
    /// Copies into `scalar`'s existing HBM allocation instead of a fresh one.
    pub fn output<'h>(self, scalar: &'h mut HbmScalar<D, B>) -> Prepared<FromHost<'t, D, B, &'h mut HbmScalar<D, B>>> {
        let FromHost { value, dma, .. } = self.into_work();
        Prepared::new(FromHost { value, dma, to: scalar })
    }
}
