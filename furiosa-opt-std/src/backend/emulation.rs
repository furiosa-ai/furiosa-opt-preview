use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::scalar::Scalar;
use crate::storage::BufStorage;
use crate::tensor::memory::{Address, HbmTensor, HostTensor};

use crate::backend::Backend;

/// Emulation backend: host-side buffer interpreter using `BufStorage` storage.
///
/// Runs every operation over the physical staging buffer through `BufStorage`'s inherent
/// methods, with no device behind it. This is the buffer/array-semantics counterpart to
/// `Simulation`'s canonical `MathStorage` interpretation, over the same `Vec<D>` staging buffer
/// shape `Npu` uses. DMA is the shared host-side `transpose` passthrough.
#[derive(Debug, Clone, Copy)]
pub struct Emulation;

impl Backend for Emulation {
    type Storage<D: Scalar> = BufStorage<D>;

    fn from_buf<D: Scalar>(mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        BufStorage::from_buf(mapping, data)
    }

    fn uninit<D: Scalar>(mapping: &MappingValue) -> Self::Storage<D> {
        // BUF has no Opt::Uninit; the blank canvas is just zeros (overwritten by the relayout that
        // allocates it; padding cells are don't-care).
        BufStorage::from_buf(mapping, std::iter::repeat_n(D::zero(), mapping.size()))
    }

    fn into_buf<D: Scalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<D> {
        storage.into_buf(mapping)
    }

    fn map<D: Scalar, D2: Scalar>(src: &Self::Storage<D>, f: impl FnMut(D) -> D2) -> Self::Storage<D2> {
        src.map(f)
    }

    fn zip_with<D: Scalar, D2: Scalar, D3: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        f: impl Fn(D, D2) -> D3,
    ) -> Self::Storage<D3> {
        a.zip_with(b, f)
    }

    fn zip3_with<D: Scalar, D2: Scalar, D3: Scalar, D4: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        c: &Self::Storage<D3>,
        f: impl Fn(D, D2, D3) -> D4,
    ) -> Self::Storage<D4> {
        a.zip3_with(b, c, f)
    }

    fn transpose<D: Scalar, Src: M, Dst: M>(
        dst: &mut Self::Storage<D>,
        src: &Self::Storage<D>,
        src_offset: &Index,
        dst_offset: &Index,
        src_map: &MappingValue,
        dst_map: &MappingValue,
        allow_broadcast: bool,
    ) {
        dst.transpose::<Src, Dst>(src, src_offset, dst_offset, src_map, dst_map, allow_broadcast);
    }

    fn reduce<D: Scalar, Src: M, Dst: M>(
        src: &Self::Storage<D>,
        reduce_fn: impl Fn(D, D) -> D,
        identity: D,
        allow_broadcast: bool,
    ) -> Self::Storage<D> {
        src.reduce::<Src, Dst, _>(reduce_fn, identity, allow_broadcast)
    }

    fn scatter<D: Scalar, Src: M, Key: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    ) {
        src.scatter::<Src, Key, Dst, Idx>(dst, index, scaled);
    }

    fn gather<D: Scalar, Src: M, Dst: M, Idx: M>(
        src: &Self::Storage<D>,
        dst: &mut Self::Storage<D>,
        index: &Self::Storage<i32>,
        scaled: bool,
    ) {
        src.gather::<Src, Dst, Idx>(dst, index, scaled);
    }

    fn reshape<D: Scalar, Src: M, Dst: M>(src: &Self::Storage<D>) -> Self::Storage<D> {
        src.reshape::<Src, Dst>()
    }

    fn transmute<D: Scalar, Src: M, Dst: M>(
        storage: Self::Storage<D>,
        src_map: &MappingValue,
        dst_map: &MappingValue,
    ) -> Self::Storage<D> {
        storage.transmute(src_map, dst_map)
    }

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
        address: Address,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        HbmTensor::new(host.inner().transpose(true), address)
    }

    async fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        hbm.inner().transpose(true).into()
    }
}
