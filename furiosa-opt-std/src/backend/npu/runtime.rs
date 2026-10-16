use std::ops::RangeBounds;
use std::sync::Arc;

use furiosa_opt_rt as rt;

use super::backend::{Device, Npu};
use super::registry;
use crate::Error;
use crate::backend::{Allocation, Runtime};
use crate::runtime::{
    Access, DeviceOutput, DeviceSend, HbmBufferSlice, HbmBufferSliceMut, HostBufferSlice, HostBufferSliceMut, Launched,
    Placement, Region, Submitted, Topology, Usage, launch_memory,
};

impl Runtime for Npu {
    type Device = Arc<Device>;
    type Kernel = usize;
    type Buffer = rt::Buffer;

    fn open(placements: Vec<Placement>) -> Result<Self::Device, Error> {
        Ok(Arc::new(Device {
            device: Arc::new(rt::Device::new(placements.as_slice())?),
            registry: registry::Registry::default(),
        }))
    }

    fn placement(device: &Self::Device) -> &Placement {
        device.device.placement()
    }

    fn topology(device: &Self::Device) -> Topology {
        device.device.topology()
    }

    fn capacity(device: &Self::Device) -> usize {
        device.device.capacity(rt::memory::Kind::DramShared)
    }

    fn usage(device: &Self::Device) -> Usage {
        device.device.usage(rt::memory::Kind::DramShared)
    }

    fn alloc(device: &Self::Device, bytes: usize) -> Result<rt::Buffer, Error> {
        // Multi-chip kernels address their IO through peers; one chip's shared range is all DRAM.
        Ok(device.device.alloc(rt::memory::Kind::DramShared, bytes)?)
    }

    async fn write<'a>(
        device: &Self::Device,
        pairs: Vec<(HostBufferSlice<'a>, HbmBufferSliceMut<'a, Self>)>,
    ) -> Result<(), Error> {
        let mut pieces = Vec::with_capacity(pairs.len());
        for (host, to) in pairs {
            let Region { chips, bytes } = to.region;
            let len = bytes.len();
            let views = to.buffer.slice_mut(bytes)?.chips().enumerate();
            let views = views.filter(|(rank, _)| chips & 1 << rank != 0).map(|(_, view)| view);
            pieces.extend(host.0.chunks(len.max(1)).zip(views));
        }
        Ok(device.device.write(pieces).await?)
    }

    async fn read(
        device: &Self::Device,
        from: HbmBufferSlice<'_, Self>,
        host: HostBufferSliceMut<'_>,
    ) -> Result<(), Error> {
        let Region { chips, bytes } = from.region;
        let len = bytes.len();
        let host = host.0;
        let views = from.buffer.slice(bytes)?.chips().enumerate();
        let views = views.filter(|(rank, _)| chips & 1 << rank != 0).map(|(_, view)| view);
        Ok(device.device.read(views.zip(host.chunks(len))).await?)
    }

    fn synchronize(device: &Self::Device) -> impl Future<Output = Result<(), Error>> + Send + use<> {
        let synchronized = device.device.synchronize();
        async move { Ok(synchronized.await?) }
    }

    fn load(device: &Self::Device, path: &'static str, key: rt::registry::Key<'_>) -> Result<usize, Error> {
        let index = registry::index(path, key)?;
        registry::function(device, index)?;
        Ok(index)
    }

    fn launch<Args: DeviceSend<Self>, Output: DeviceOutput<Self>>(
        device: &mut crate::Device<Self>,
        kernel: usize,
        args: Args,
        output: &mut Output,
        _: impl FnOnce(&mut crate::Device<Self>, Args) -> Output,
    ) -> Submitted<Launched<()>> {
        Submitted::launched((|| {
            let function = registry::function(device.pdma.device(), kernel)?;
            args.prepare(device.pdma.device())?;
            output.prepare(device.pdma.device())?;
            // Each tensor's memory binds as a view of its bytes on every chip.
            let arguments = launch_memory(&args, output)?.map(|(memory, access)| {
                let view = memory
                    .buffer
                    .slice(memory.region.bytes)
                    .expect("a slice lies inside its buffer");
                match access {
                    Access::Read => rt::Argument::Read(view),
                    Access::Write => rt::Argument::Write(view),
                }
            });
            let held = launch_memory(&args, output)?
                .flat_map(|(slice, access)| {
                    slice
                        .behind
                        .iter()
                        .map(move |buffer| buffer.slice(..).map(|view| (view, access)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(function.launch_holding(arguments, &held)?))
        })())
    }
}

impl Allocation for rt::Buffer {
    fn size(&self) -> usize {
        rt::Buffer::len(self)
    }

    fn chips(&self) -> usize {
        self.topology().chips.into()
    }

    fn addr(&self) -> usize {
        rt::Buffer::addr(self)
    }

    fn split_at(self, offset: usize) -> Result<(Self, Self), Error> {
        Ok(rt::Buffer::split_at(self, offset)?)
    }

    fn alias(&self, bytes: impl RangeBounds<usize>) -> Result<Self, Error> {
        Ok(rt::Buffer::alias(self, bytes)?)
    }

    fn join(&mut self, other: &Self) -> bool {
        rt::Buffer::join(self, other)
    }

    fn ready(&self) -> impl Future<Output = Result<(), Error>> + Send + use<> {
        let ready = rt::Buffer::ready(self, rt::Access::Read);
        async move { Ok(ready.await?) }
    }
}
