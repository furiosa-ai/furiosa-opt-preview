use furiosa_mapping::*;

use crate::Error;
use crate::scalar::{MaterializableScalar, RuntimeScalar, Scalar};
use crate::storage::BufStorage;
use crate::tensor::Tensor;
use crate::tensor::memory::{DmAllocError, DmTensor, HbmScalar, HbmTensor};

use crate::backend::indirect::IndexUnit;
use crate::backend::{Runtime, internal};
use crate::cast::{ContractionAccumulator, ContractionCast};
use crate::context::{Dma, DmaContext};
use crate::runtime::{
    Access, DeviceOutput, DeviceSend, HbmBufferSlice, HbmBufferSliceMut, HostBuffer, HostBufferSlice,
    HostBufferSliceMut, Launched, Placement, Submitted, Topology, Usage, launch_memory,
};

mod buffer;
use buffer::{Buffer, Bytes, Capacity};
use furiosa_opt_rt::memory::Kind;
use std::sync::Arc;

/// Host backend: interprets device functions on the host, over host memory that stands in for
/// device memory with the device's budget, aliases and access conflicts.
#[derive(Debug, Clone, Copy)]
pub struct Cpu;

/// The host standing in for a device of `topology`, with its memory budget.
#[derive(Clone, Debug)]
pub struct Host {
    topology: Topology,
    placement: Placement,
    capacity: Arc<Capacity>,
}

impl Runtime for Cpu {
    type Device = Host;

    /// The host backend runs the function's Rust body, so a kernel names nothing.
    type Kernel = ();

    type Buffer = Buffer;

    /// Memory of the DRAM an RNGD chip offers the topology's PEs before the firmware's share: an
    /// upper bound of the NPU's, which learns its free range when the device boots.
    fn open(placements: Vec<Placement>) -> Result<Host, Error> {
        use furiosa_opt_abi::arch::rngd::{DRAM, PES};
        let placement = placements
            .into_iter()
            .next()
            .ok_or(Error::Invalid("no placement candidates"))?;
        let topology = Topology {
            chips: u8::try_from(placement.chips.len()).map_err(|_| Error::Invalid("too many placement chips"))?,
            pes: u8::try_from(placement.pes.len()).map_err(|_| Error::Invalid("too many placement PEs"))?,
        };
        let dram = ((DRAM / u64::from(PES)) as usize & !4095) * usize::from(topology.pes);
        Ok(Host {
            topology,
            placement,
            capacity: Arc::new(Capacity::new(dram)),
        })
    }

    fn placement(device: &Host) -> &Placement {
        &device.placement
    }

    fn topology(device: &Host) -> Topology {
        device.topology
    }

    fn capacity(device: &Host) -> usize {
        device.capacity.capacity(Kind::DramShared)
    }

    fn usage(device: &Host) -> Usage {
        device.capacity.usage(Kind::DramShared)
    }

    fn alloc(device: &Host, bytes: usize) -> Result<Buffer, Error> {
        Ok(Buffer::alloc(
            &device.capacity,
            Kind::DramShared,
            usize::from(device.topology.chips),
            bytes,
        )?)
    }

    async fn write<'a>(
        device: &Host,
        pairs: Vec<(HostBufferSlice<'a>, HbmBufferSliceMut<'a, Self>)>,
    ) -> Result<(), Error> {
        for (_, to) in &pairs {
            to.buffer.check_device(&device.capacity)?;
        }
        for (host, to) in pairs {
            to.buffer.write(&device.capacity, &to.region, host.0.bytes())?;
        }
        Ok(())
    }

    async fn read(device: &Host, from: HbmBufferSlice<'_, Self>, host: HostBufferSliceMut<'_>) -> Result<(), Error> {
        from.buffer.read(&device.capacity, &from.region, host.0.into_bytes())
    }

    /// A host launch finishes before it returns, and so does a host transfer.
    fn synchronize(_: &Host) -> impl Future<Output = Result<(), Error>> + Send + use<> {
        std::future::ready(Ok(()))
    }

    /// The host backend runs the function's Rust body and loads nothing.
    fn load(_: &Host, _: &'static str, _: furiosa_opt_rt::registry::Key<'_>) -> Result<(), Error> {
        Ok(())
    }

    /// Runs the function's Rust body over its memory, refused as a device refuses it, and copies
    /// its returns into `output`. The launch finishes before it returns, so only its own uses can
    /// conflict.
    fn launch<Args: DeviceSend<Self>, Output: DeviceOutput<Self>>(
        device: &mut crate::Device<Self>,
        (): (),
        args: Args,
        output: &mut Output,
        body: impl FnOnce(&mut crate::Device<Self>, Args) -> Output,
    ) -> Submitted<Launched<()>> {
        Submitted::launched((|| {
            args.prepare(device.pdma.device())?;
            output.prepare(device.pdma.device())?;
            let capacity = &device.pdma.device().capacity;
            let uses = launch_memory(&args, output)?
                .flat_map(|(slice, access)| {
                    std::iter::once((slice.buffer, slice.region, access)).chain(
                        slice
                            .behind
                            .iter()
                            .map(move |buffer| (buffer, HbmBufferSlice::<Self>::whole(buffer).region, access)),
                    )
                })
                .collect::<Vec<_>>();
            for (memory, _, _) in &uses {
                memory.check_device(capacity)?;
            }
            buffer::check_conflicts(uses.iter().map(|(memory, region, access)| (*memory, region, *access)))?;
            let returned = body(device, args);
            for ((to, _), (from, _)) in output.arguments(Access::Write)?.zip(returned.arguments(Access::Read)?) {
                to.buffer.store(&to.region, &from.buffer.load(&from.region));
            }
            Ok(None)
        })())
    }
}

impl internal::Backend for Cpu {
    type Storage<D: Scalar> = BufStorage<D, Bytes>;

    fn load_memory<D: Scalar>(memory: &HbmBufferSlice<'_, Self>) -> Self::Storage<D> {
        memory.buffer.load(&memory.region).into()
    }

    fn scratch(chips: usize, bytes: usize) -> Buffer {
        Buffer::scratch(chips, bytes)
    }

    fn from_vec<D: Scalar>(_mapping: &Mapping, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        BufStorage::from_vec(data)
    }

    fn from_buf<D: MaterializableScalar>(_mapping: &Mapping, buf: Vec<u8>) -> Self::Storage<D> {
        BufStorage::from_buf(buf)
    }

    /// A resident's memory, which no other owner names; its values stay in the tensor.
    fn alloc_dm<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        group: &Host,
    ) -> Result<DmTensor<D, Chip, Cluster, Slice, Element, Self>, DmAllocError> {
        let devices = usize::from(group.topology.chips);
        if Chip::SIZE != devices {
            return Err(DmAllocError::Topology {
                tensor_chips: Chip::SIZE,
                device_chips: devices,
            });
        }
        let bytes = D::size_in_bytes_from_length(Element::SIZE);
        let buffer = Buffer::alloc(&group.capacity, Kind::Sram, devices, bytes)?;
        let mut tensor = DmTensor::from_parts(Tensor::zeroed());
        tensor.own(buffer);
        Ok(tensor)
    }

    fn zeroed<D: Scalar>(mapping: &Mapping) -> Self::Storage<D> {
        BufStorage::zeroed(mapping.size())
    }

    fn into_vec<D: MaterializableScalar>(storage: Self::Storage<D>, mapping: &Mapping) -> Vec<D> {
        storage.into_vec(mapping)
    }

    fn into_buf<D: Scalar>(storage: Self::Storage<D>, mapping: &Mapping) -> Vec<u8> {
        storage.into_buf(mapping)
    }

    fn map<D: Scalar, D2: Scalar>(src: &Self::Storage<D>, f: impl Fn(D) -> D2 + Sync) -> Self::Storage<D2> {
        src.map(f)
    }

    fn zip_with<D: MaterializableScalar, D2: MaterializableScalar, D3: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> Self::Storage<D3> {
        a.zip_with(b, f)
    }

    fn zip3_with<D: MaterializableScalar, D2: MaterializableScalar, D3: MaterializableScalar, D4: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        c: &Self::Storage<D3>,
        f: impl Fn(D, D2, D3) -> D4 + Sync,
    ) -> Self::Storage<D4> {
        a.zip3_with(b, c, f)
    }

    fn transpose<D: Scalar, Src: M, Dst: M>(
        dst: &mut Self::Storage<D>,
        src: &Self::Storage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &Mapping,
        dst_map: &Mapping,
        allow_broadcast: bool,
        preserve_copies: bool,
    ) {
        dst.transpose::<Src, Dst>(
            src,
            src_offset,
            dst_offset,
            src_map,
            dst_map,
            allow_broadcast,
            preserve_copies,
        );
    }

    fn reduce<D: MaterializableScalar, Src: M, Dst: M>(
        src: &Self::Storage<D>,
        reduce_fn: impl Fn(D, D) -> D + Sync,
        identity: D,
        allow_broadcast: bool,
    ) -> Self::Storage<D> {
        src.reduce::<Src, Dst, _>(reduce_fn, identity, allow_broadcast)
    }

    fn contraction<D: ContractionCast + MaterializableScalar>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &Mapping,
        rhs_map: &Mapping,
        pre_reduce: &Mapping,
        out: &Mapping,
    ) -> Self::Storage<D> {
        // `BufStorage` is a bare buffer with no layout of its own, so it reads operand strides from
        // `lhs_map`/`rhs_map` (the same reason its `transpose` above takes `src_map`/`dst_map`).
        BufStorage::contraction(lhs, rhs, lhs_map, rhs_map, pre_reduce, out)
    }

    fn contraction_prewidened<D: ContractionAccumulator>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_map: &Mapping,
        rhs_map: &Mapping,
        pre_reduce: &Mapping,
        out: &Mapping,
    ) -> Self::Storage<D> {
        // `BufStorage` is a bare buffer with no layout of its own, so it reads operand strides from
        // `lhs_map`/`rhs_map` (the same reason its `transpose` above takes `src_map`/`dst_map`).
        BufStorage::contraction_prewidened(lhs, rhs, lhs_map, rhs_map, pre_reduce, out)
    }

    fn scatter<D: Scalar, Src: M, Domain: M, IndexedAxis: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        chip: &Mapping,
        unit: IndexUnit,
    ) {
        src.scatter::<Src, Domain, IndexedAxis, Dst, Idx>(dst, index, chip, unit);
    }

    fn gather<D: MaterializableScalar, Src: M, IndexedAxis: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        chip: &Mapping,
        read: crate::backend::GatherRead<'_, Self::Storage<i32>>,
    ) {
        src.gather::<Src, IndexedAxis, Dst, Idx>(dst, index, chip, read);
    }

    fn gather_table<D: MaterializableScalar, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        mapping: &Mapping,
        indexed: &Mapping,
        chip: &Mapping,
        read: crate::backend::GatherRead<'_, Self::Storage<i32>>,
    ) {
        src.gather_into::<Dst, Idx>(dst, index, mapping, indexed, chip, read);
    }

    fn scatter_table<D: Scalar, Src: M, Key: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        mapping: &Mapping,
        indexed: &Mapping,
        chip: &Mapping,
        unit: IndexUnit,
    ) {
        src.scatter_into::<Src, Key, Idx>(dst, index, mapping, indexed, chip, unit);
    }

    fn reshape<D: Scalar, Src: M, Dst: M>(src: &Self::Storage<D>) -> Self::Storage<D> {
        src.reshape::<Src, Dst>()
    }

    fn transmute<D: MaterializableScalar, Src: M, Dst: M>(
        storage: Self::Storage<D>,
        src_map: &Mapping,
        dst_map: &Mapping,
    ) -> Self::Storage<D> {
        storage.transmute(src_map, dst_map)
    }

    async fn to_hbm_into<D: MaterializableScalar, Chip: M, Element: M>(
        host: &Self::Storage<D>,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
        hbm: &mut HbmTensor<D, Chip, Element, Self>,
    ) -> Result<(), Error> {
        let memory = hbm.memory_mut();
        memory
            .buffer
            .write(&dma.device().capacity, &memory.region, host.as_bytes())
    }

    async fn to_hbm_scalar_into<D: RuntimeScalar>(
        value: D,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
        hbm: &mut HbmScalar<D, Self>,
    ) -> Result<(), Error> {
        let memory = hbm.memory_mut();
        let bytes = HbmScalar::<D, Self>::encode(value).repeat(memory.region.chips.count_ones() as usize);
        memory.buffer.write(&dma.device().capacity, &memory.region, &bytes)
    }

    async fn from_hbm_into<D: MaterializableScalar, Chip: M, Element: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
        dma: &DmaContext<{ Dma::Pcie }, Self>,
        host: &mut Self::Storage<D>,
    ) -> Result<(), Error> {
        let memory = hbm.memory();
        let mut bytes = vec![0; memory.region.host_len()];
        memory.buffer.read(&dma.device().capacity, &memory.region, &mut bytes)?;
        *host = BufStorage::from_buf(bytes);
        Ok(())
    }

    /// Host memory moves without a copy unless it is page-locked, whose bytes are copied out.
    fn host_storage<D: Scalar>(buffer: HostBuffer) -> Self::Storage<D> {
        BufStorage::from_buf(buffer.into())
    }

    fn host_buffer<D: Scalar>(storage: Self::Storage<D>) -> HostBuffer {
        Vec::<u8>::from(storage.into_inner()).into()
    }

    fn pin<D: MaterializableScalar>(storage: Self::Storage<D>) -> Result<Self::Storage<D>, Error> {
        Ok(storage)
    }
}
