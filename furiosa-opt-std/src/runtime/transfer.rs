//! Untyped host-HBM copies: `to_hbm` on host bytes and `to_host` on device bytes, as tensors move.

use std::future::Future;

use super::{
    HbmBuffer, HbmBufferSlice, HbmBufferSliceMut, HostBuffer, HostBufferSlice, HostBufferSliceMut, Kind, Prepared,
    Submitted, Work,
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
    ) -> Prepared<Write<'a, B>> {
        Prepared::new(Write {
            device: self.device(),
            pairs: pairs.into_iter().map(|(host, to)| (host.into(), to.into())).collect(),
        })
    }

    /// Copies existing HBM slices into host bytes in one runtime read, in pair order.
    /// A length mismatch in any pair refuses the whole batch before copying.
    pub fn to_host<'a>(
        &'a mut self,
        pairs: impl IntoIterator<Item = (impl Into<HbmBufferSlice<'a, B>>, impl Into<HostBufferSliceMut<'a>>)>,
    ) -> Prepared<Read<'a, B>> {
        Prepared::new(Read {
            device: self.device(),
            pairs: pairs
                .into_iter()
                .map(|(from, host)| (from.into(), host.into()))
                .collect(),
        })
    }
}

/// A batch of host-to-HBM copies, as a [`Prepared`] holds it; see [`DmaContext::to_hbm`].
#[doc(hidden)]
#[derive(Debug)]
pub struct Write<'a, B: Backend> {
    device: &'a B::Device,
    pairs: Vec<(HostBufferSlice<'a>, HbmBufferSliceMut<'a, B>)>,
}

impl<B: Backend> Work for Write<'_, B> {
    type Output = ();
    type Future = impl Future<Output = Result<(), Error>> + Send + 'static;

    fn submit(self) -> Submitted<Self::Future> {
        let matched = (self.pairs.iter()).all(|(host, to)| host.0.len() == to.region.host_len());
        let written = matched.then(|| B::write(self.device, self.pairs));
        Submitted(async move {
            written
                .ok_or(Error::Invalid("the host bytes do not match the memory on its chips"))?
                .await
        })
    }
}

/// A batch of HBM-to-host copies, as a [`Prepared`] holds it; see [`DmaContext::to_host`].
#[doc(hidden)]
#[derive(Debug)]
pub struct Read<'a, B: Backend> {
    device: &'a B::Device,
    pairs: Vec<(HbmBufferSlice<'a, B>, HostBufferSliceMut<'a>)>,
}

impl<B: Backend> Work for Read<'_, B> {
    type Output = ();
    type Future = impl Future<Output = Result<(), Error>> + Send + 'static;

    fn submit(self) -> Submitted<Self::Future> {
        let matched = (self.pairs.iter()).all(|(from, host)| host.0.len() == from.region.host_len());
        let read = matched.then(|| B::read(self.device, self.pairs));
        Submitted(async move {
            read.ok_or(Error::Invalid("the host bytes do not match the memory on its chips"))?
                .await
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

impl<B: Backend> Work for ToHbm<'_, B> {
    type Output = HbmBuffer<B>;
    type Future = impl Future<Output = Result<HbmBuffer<B>, Error>> + Send + 'static;

    fn submit(self) -> Submitted<Self::Future> {
        let device = self.dma.device();
        let chips = usize::from(B::topology(device).chips);
        let len = self.from.0.len();
        let written = if len.is_multiple_of(chips) {
            B::alloc(device, Kind::Dram, len / chips).map(|buffer| {
                let mut buffer = HbmBuffer::new(buffer);
                (B::write(device, vec![(self.from, (&mut buffer).into())]), buffer)
            })
        } else {
            Err(Error::Invalid("the host bytes do not split evenly across the chips"))
        };
        Submitted(async move {
            let (written, buffer) = written?;
            written.await?;
            Ok(buffer)
        })
    }
}

impl<'t, B: Backend> Prepared<ToHbm<'t, B>> {
    /// Copies into `to`, a buffer or a [`HbmBuffer::slice_mut`] of one, instead of a fresh buffer.
    pub fn output(self, to: impl Into<HbmBufferSliceMut<'t, B>>) -> Prepared<Write<'t, B>> {
        let ToHbm { from, dma } = self.into_work();
        dma.to_hbm([(from, to.into())])
    }
}

/// An HBM-to-host copy of bytes, as a [`Prepared`] holds it; see [`HbmBufferSlice::to_host`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ToHost<'t, B: Backend> {
    from: HbmBufferSlice<'t, B>,
    dma: &'t mut DmaContext<{ Dma::Pcie }, B>,
}

impl<B: Backend> Work for ToHost<'_, B> {
    type Output = HostBuffer;
    type Future = impl Future<Output = Result<HostBuffer, Error>> + Send + 'static;

    fn submit(self) -> Submitted<Self::Future> {
        let mut host = HostBuffer::zero(self.from.region.host_len());
        let read = B::read(self.dma.device(), vec![(self.from, (&mut host).into())]);
        Submitted(async move {
            read.await?;
            Ok(host)
        })
    }
}

impl<'t, B: Backend> Prepared<ToHost<'t, B>> {
    /// Copies into `to`, host bytes or a [`HostBuffer`], instead of a fresh [`HostBuffer`].
    pub fn output(self, to: impl Into<HostBufferSliceMut<'t>>) -> Prepared<Read<'t, B>> {
        let ToHost { from, dma } = self.into_work();
        dma.to_host([(from, to.into())])
    }
}
