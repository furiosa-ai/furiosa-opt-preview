use furiosa_mapping::Mapping as MappingValue;
use furiosa_mapping::*;

use crate::cast::ContractionCast;
use crate::scalar::Scalar;
use crate::storage::MathStorage;
use crate::tensor::memory::{HbmTensor, HostTensor};

use crate::backend::Backend;

/// Simulation backend: mapping-expression-based interpretation.
///
/// All per-operation behavior lives on [`MathStorage`]'s inherent methods; DMA is the shared
/// host-side `transpose` passthrough.
#[derive(Debug, Clone, Copy)]
pub struct Simulation;

impl Backend for Simulation {
    type Storage<D: Scalar> = MathStorage<D>;

    fn from_buf<D: Scalar>(mapping: &MappingValue, data: impl IntoIterator<Item = D>) -> Self::Storage<D> {
        MathStorage::from_buf(mapping, data)
    }

    fn uninit<D: Scalar>(mapping: &MappingValue) -> Self::Storage<D> {
        MathStorage::uninit(mapping)
    }

    fn into_buf<D: Scalar>(storage: Self::Storage<D>, mapping: &MappingValue) -> Vec<D> {
        storage.into_buf(mapping)
    }

    fn map<D: Scalar, D2: Scalar>(src: &Self::Storage<D>, f: impl Fn(D) -> D2 + Sync) -> Self::Storage<D2> {
        src.map(f)
    }

    fn zip_with<D: Scalar, D2: Scalar, D3: Scalar>(
        a: &Self::Storage<D>,
        b: &Self::Storage<D2>,
        f: impl Fn(D, D2) -> D3 + Sync,
    ) -> Self::Storage<D3> {
        a.zip_with(b, f)
    }

    fn zip3_with<D: Scalar, D2: Scalar, D3: Scalar, D4: Scalar>(
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
        _src_map: &MappingValue,
        _dst_map: &MappingValue,
        allow_broadcast: bool,
    ) {
        // `MathStorage` already carries its live-axis mapping (its `axes`), so it ignores `*_map`.
        dst.transpose::<Src, Dst>(src, src_offset, dst_offset, allow_broadcast);
    }

    fn reduce<D: Scalar, Src: M, Dst: M>(
        src: &Self::Storage<D>,
        reduce_fn: impl Fn(D, D) -> D + Sync,
        identity: D,
        allow_broadcast: bool,
    ) -> Self::Storage<D> {
        src.reduce::<Src, Dst, _>(reduce_fn, identity, allow_broadcast)
    }

    fn contraction<D: Scalar + ContractionCast, Union: M, Lhs: M, Rhs: M, Out: M>(
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
    ) -> Self::Storage<D> {
        // `MathStorage` reads each operand's axes from its storage, so it needs only `Union` / `Out`.
        MathStorage::contraction::<Union, Out>(lhs, rhs)
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
        _src_map: &MappingValue,
        _dst_map: &MappingValue,
    ) -> Self::Storage<D> {
        // `MathStorage` is canonical (padding-free), so a size-shrinking relabel needs no data move.
        storage
    }

    async fn to_hbm<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Self>,
    ) -> HbmTensor<D, Chip, Element2, Self> {
        HbmTensor::new(host.inner().transpose(true), 0)
    }

    async fn from_hbm<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Self>,
    ) -> HostTensor<D, Element2, Self> {
        hbm.inner().transpose(true).into()
    }
}
