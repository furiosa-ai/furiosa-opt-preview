use std::sync::Arc;

use furiosa_mapping::*;

use crate::Error;
use crate::scalar::{MaterializableScalar, Scalar};
use crate::storage::BufStorage;
use crate::tensor::Tensor;
use crate::tensor::memory::{DmAllocError, DmTensor};

use super::registry;
use crate::backend::indirect::IndexUnit;
use crate::backend::internal;
use crate::cast::{ContractionAccumulator, ContractionCast};
use crate::runtime::{HostBuffer, HostBufferSlice, HostBufferSliceMut};
use furiosa_opt_rt as rt;

/// NPU backend.
///
/// Host storage uses packed CPU memory; an HBM tensor's bytes live on the device and its host
/// side holds none.
#[derive(Debug, Clone, Copy)]
pub struct Npu;

/// An opened device with one slot per registry entry, so a `#[device]` fn loads once per device.
pub struct Device {
    pub(super) device: Arc<rt::Device>,
    pub(super) registry: registry::Registry,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Device").finish_non_exhaustive()
    }
}

impl<D: Scalar> From<BufStorage<D, HostBuffer>> for HostBuffer {
    fn from(storage: BufStorage<D, HostBuffer>) -> Self {
        storage.into_inner()
    }
}

impl internal::Backend for Npu {
    type Storage<D: Scalar> = BufStorage<D, HostBuffer>;

    fn scratch(_: usize, _: usize) -> rt::Buffer {
        panic!("the NPU host reaches device memory only by DMA, not a device function's body")
    }

    fn from_vec<D: Scalar>(_mapping: &Mapping, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        BufStorage::from_vec(data)
    }

    fn from_buf<D: MaterializableScalar>(_mapping: &Mapping, buf: Vec<u8>) -> Self::Storage<D> {
        BufStorage::from_buf(buf)
    }

    fn alloc_dm<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M>(
        group: &Self::Device,
    ) -> Result<DmTensor<D, Chip, Cluster, Slice, Element, Self>, DmAllocError> {
        // One slice's bytes: a resident sits at the same address in every slice of every device.
        let size = D::size_in_bytes_from_length(Element::SIZE);
        let devices = usize::from(group.device.topology().chips);
        if Chip::SIZE != devices {
            return Err(DmAllocError::Topology {
                tensor_chips: Chip::SIZE,
                device_chips: devices,
            });
        }
        let buffer = group.device.alloc(rt::memory::Kind::Sram, size)?;
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
    ) {
        dst.transpose::<Src, Dst>(src, src_offset, dst_offset, src_map, dst_map, allow_broadcast);
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
        domain: &Mapping,
        chip: &Mapping,
        prefix: Option<usize>,
        unit: IndexUnit,
    ) {
        src.gather::<Src, IndexedAxis, Dst, Idx>(dst, index, domain, chip, prefix, unit);
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

    fn bytes<D: Scalar>(storage: &Self::Storage<D>) -> HostBufferSlice<'_> {
        storage.inner().into()
    }

    fn bytes_mut<D: Scalar>(storage: &mut Self::Storage<D>) -> HostBufferSliceMut<'_> {
        storage.inner_mut().into()
    }

    fn pin<D: MaterializableScalar>(storage: Self::Storage<D>) -> Result<Self::Storage<D>, Error> {
        Ok(BufStorage::from_inner(storage.into_inner().pin()?))
    }
}
