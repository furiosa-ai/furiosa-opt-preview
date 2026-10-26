//! Untyped host-HBM copies: `to_hbm` on host bytes and `to_host` on device bytes, as tensors move.

use std::future::Future;

use super::{
    HbmBuffer, HbmBufferSlice, HbmBufferSliceMut, HostBuffer, HostBufferSlice, HostBufferSliceMut, Issued, Kind,
    Prepared, Submitted, Work,
};
use crate::Error;
use crate::backend::Backend;
use crate::context::{Dma, DmaContext};

impl<B: Backend> DmaContext<{ Dma::Pcie }, B> {
    /// Copies host bytes into existing HBM slices in one runtime write, in pair order.
    /// A length mismatch in any pair refuses the whole batch before copying.
    pub fn to_hbm<'a>(
        &'a mut self,
        pairs: impl IntoIterator<Item = (impl Into<HostBufferSlice<'a>>, impl Into<HbmBufferSliceMut<'a, B>>)>,
    ) -> Prepared<impl Future<Output = Result<(), Error>>> {
        let pairs = pairs
            .into_iter()
            .map(|(host, to)| (host.into(), to.into()))
            .collect::<Vec<_>>();
        Prepared::new(async move {
            if pairs
                .iter()
                .any(|(host, to)| host.0.bytes().len() != to.region.host_len())
            {
                return Err(Error::Invalid("the host bytes do not match the memory on its chips"));
            }
            B::write(self.device(), pairs).await
        })
    }
}

impl HostBuffer {
    /// Prepares a copy of these bytes to HBM; see [`HostBufferSlice::to_hbm`].
    pub fn to_hbm<'t, B: Backend>(&'t self, dma: &'t mut DmaContext<{ Dma::Pcie }, B>) -> Prepared<ToHbm<'t, B>> {
        HostBufferSlice::from(self).to_hbm(dma)
    }
}

impl<'t> HostBufferSlice<'t> {
    /// Prepares a copy of these bytes to HBM, once the device work submitted before it on the
    /// destination ended: `.await` copies them into a fresh buffer that holds an
    /// equal share on every chip, `.output(to)` into existing memory whose chips hold exactly these
    /// bytes, chip after chip in rank order.
    pub fn to_hbm<B: Backend>(self, dma: &'t mut DmaContext<{ Dma::Pcie }, B>) -> Prepared<ToHbm<'t, B>> {
        Prepared::new(ToHbm { from: self, dma })
    }
}

impl<B: Backend> HbmBuffer<B> {
    /// Prepares a copy of this buffer to the host; see [`HbmBufferSlice::to_host`].
    pub fn to_host<'t>(&'t self, dma: &'t mut DmaContext<{ Dma::Pcie }, B>) -> Prepared<ToHost<'t, B>> {
        HbmBufferSlice::from(self).to_host(dma)
    }
}

impl<'t, B: Backend> HbmBufferSlice<'t, B> {
    /// Prepares a copy of this memory to the host, chip after chip in rank order, once the device
    /// work submitted before it that writes the memory ended: `.await` copies it into a fresh
    /// [`HostBuffer`], `.output(to)` into host bytes of exactly its length.
    pub fn to_host(self, dma: &'t mut DmaContext<{ Dma::Pcie }, B>) -> Prepared<ToHost<'t, B>> {
        Prepared::new(ToHost { from: self, dma })
    }
}

/// A host-to-HBM copy of bytes, as a [`Prepared`] holds it; see [`HostBufferSlice::to_hbm`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ToHbm<'t, B: Backend> {
    from: HostBufferSlice<'t>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
}

impl<B: Backend> ToHbm<'_, B> {
    async fn write(self, to: HbmBufferSliceMut<'_, B>) -> Result<(), Error> {
        self.dma.to_hbm([(self.from, to)]).await
    }
}

impl<'t, B: Backend> Work for ToHbm<'t, B> {
    type Output = HbmBuffer<B>;
    type Future = Issued<impl Future<Output = Result<HbmBuffer<B>, Error>>>;

    fn submit(self) -> Submitted<Self::Future> {
        async move {
            let chips = usize::from(B::topology(self.dma.device()).chips);
            let len = self.from.0.bytes().len();
            if !len.is_multiple_of(chips) {
                return Err(Error::Invalid("the host bytes do not split evenly across the chips"));
            }
            let mut buffer = HbmBuffer(B::alloc(self.dma.device(), Kind::Dram, len / chips)?);
            self.write(HbmBufferSliceMut::from(&mut buffer)).await?;
            Ok(buffer)
        }
        .submit()
    }
}

impl<'t, B: Backend> Prepared<ToHbm<'t, B>> {
    /// Copies into `to`, a buffer or a [`HbmBuffer::slice_mut`] of one, instead of a fresh buffer.
    pub fn output(self, to: impl Into<HbmBufferSliceMut<'t, B>>) -> Prepared<impl Future<Output = Result<(), Error>>> {
        Prepared::new(self.into_work().write(to.into()))
    }
}

/// An HBM-to-host copy of bytes, as a [`Prepared`] holds it; see [`HbmBufferSlice::to_host`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ToHost<'t, B: Backend> {
    from: HbmBufferSlice<'t, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
}

impl<B: Backend> ToHost<'_, B> {
    async fn read(self, to: HostBufferSliceMut<'_>) -> Result<(), Error> {
        if to.0.len() != self.from.region.host_len() {
            return Err(Error::Invalid("the host bytes do not match the memory on its chips"));
        }
        B::read(self.dma.device(), self.from, to).await
    }
}

impl<'t, B: Backend> Work for ToHost<'t, B> {
    type Output = HostBuffer;
    type Future = Issued<impl Future<Output = Result<HostBuffer, Error>>>;

    fn submit(self) -> Submitted<Self::Future> {
        async move {
            let mut host = HostBuffer::zero(self.from.region.host_len());
            self.read(HostBufferSliceMut::from(&mut host)).await?;
            Ok(host)
        }
        .submit()
    }
}

impl<'t, B: Backend> Prepared<ToHost<'t, B>> {
    /// Copies into `to`, host bytes or a [`HostBuffer`], instead of a fresh [`HostBuffer`].
    pub fn output(self, to: impl Into<HostBufferSliceMut<'t>>) -> Prepared<impl Future<Output = Result<(), Error>>> {
        Prepared::new(self.into_work().read(to.into()))
    }
}
