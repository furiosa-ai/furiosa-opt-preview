//! SPM tensors with scalar access on PE 0 and borrowed regions for DMA.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_lower::validate_spm_scalar_pe;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::constraints;
use crate::runtime::CurrentBackend;
use crate::scalar::{RuntimeIndex, RuntimeScalar, Scalar};
use crate::spm_scalar::SpmScalar;
use crate::tensor::Tensor;
use crate::tensor::view::{TensorView, TensorViewMut, assert_tile};

/// Tensor stored in SPM, with a payload per active chip and cluster.
/// `Pe` distributes the payload across PEs; `Element` describes each PE's elements.
#[primitive(SpmTensor)]
#[derive(Debug)]
pub struct SpmTensor<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend = CurrentBackend> {
    pub(super) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Pe, Element>>>, B>,
    _marker: PhantomData<(D, Chip, Cluster, Pe, Element)>,
}

impl<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend> SpmTensor<D, Chip, Cluster, Pe, Element, B> {
    /// Logical SPM mapping, including the explicit fused PE axis.
    pub type Mapping = m![{ Chip }, { Cluster }, { Pe }, { Element }];

    /// Storage backing all chip and cluster payloads.
    pub(crate) fn storage(&self) -> &B::Storage<D> {
        &self.inner.inner
    }

    /// See [`Self::storage`].
    pub(crate) fn storage_mut(&mut self) -> &mut B::Storage<D> {
        &mut self.inner.inner
    }

    /// Constructs an SPM tensor after checking, at compile time, that `Cluster` and `Pe` name
    /// hardware counts and that `Element` fits the scratchpad one PE holds. The over-budget case is
    /// a `compile_fail` doctest in `furiosa-opt-examples`' `negative::scatter_gather`.
    pub(crate) fn new(inner: Tensor<D, Self::Mapping, B>) -> Self {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_pe_size::<Pe>();
        constraints::assert_spm_capacity::<D, Element>();
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<D: Scalar + RuntimeScalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend>
    SpmTensor<D, Chip, Cluster, Pe, Element, B>
{
    /// Reads PE 0 at the same `index` in each active chip and cluster.
    /// Requires `index < len()`; the device does not check runtime indices.
    #[primitive(SpmTensor::get)]
    pub fn get(&self, index: impl RuntimeIndex) -> SpmScalar<D, Chip, Cluster, B> {
        assert_single_live_pe::<Pe>();
        // Read padded chip and cluster slots too, keeping the host traversal symmetric.
        let stride = Pe::SIZE * Element::SIZE;
        let offset = index.into_usize();
        assert_in_payload::<Element>(offset);
        SpmScalar::from_vec(
            (0..Chip::SIZE * Cluster::SIZE).map(|cell| B::spm_local_load(self.storage(), cell * stride + offset)),
        )
    }

    /// Creates a scratchpad tensor with a copy of `values` on PE 0 in each active chip and cluster.
    #[primitive(SpmTensor::from_const)]
    pub fn from_const<const N: usize>(values: [D; N]) -> Self {
        assert_single_live_pe::<Pe>();
        assert_eq!(N, Element::SIZE, "a payload holds {} positions, got {N}", Element::SIZE);
        let stride = Pe::SIZE * Element::SIZE;
        let mut buffer = vec![D::zero(); Chip::SIZE * Cluster::SIZE * stride];
        for cell in 0..Chip::SIZE * Cluster::SIZE {
            buffer[cell * stride..cell * stride + N].copy_from_slice(&values);
        }
        Self::new(Tensor::from_vec(buffer))
    }

    /// Creates a scratchpad tensor whose logical elements initially hold zero.
    #[primitive(SpmTensor::zero)]
    pub fn zero() -> Self {
        assert_single_live_pe::<Pe>();
        Self::new(Tensor::zeroed())
    }

    /// Reinterprets the same SPM allocation under another mapping, consuming the tensor.
    ///
    /// # Safety
    ///
    /// The mappings must preserve physical element order. Equal per-level sizes alone are insufficient.
    /// Excluded PEs retain their old bytes; reshape does not initialize them.
    #[primitive(SpmTensor::reshape)]
    pub unsafe fn reshape<Chip2: M, Cluster2: M, Pe2: M, Element2: M>(
        self,
    ) -> SpmTensor<D, Chip2, Cluster2, Pe2, Element2, B> {
        crate::constraints::assert_spm_reshape_dimension_preserved::<
            Chip,
            Chip2,
            Cluster,
            Cluster2,
            Pe,
            Pe2,
            Element,
            Element2,
        >();
        SpmTensor::new(unsafe {
            self.inner
                .reshape::<m![{ Chip2 }, { Cluster2 }, { Pe2 }, { Element2 }]>()
        })
    }

    /// Number of element positions per cluster payload, including padding.
    #[allow(
        clippy::len_without_is_empty,
        reason = "`len() == 0` reads better than a method for it"
    )]
    #[inline]
    pub fn len(&self) -> u32 {
        Element::SIZE as u32
    }
}

/// Checks `index < Element::SIZE` on the host. Device runtime indices are unchecked.
pub(crate) fn assert_in_payload<Element: M>(index: usize) {
    assert!(
        index < Element::SIZE,
        "a payload holds {} positions, and this addresses {index}",
        Element::SIZE,
    );
}

/// Rejects a PE mapping with live elements outside PE 0.
pub(crate) fn assert_single_live_pe<Pe: M>() {
    validate_spm_scalar_pe(&Pe::to_value()).unwrap_or_else(|error| panic!("{error}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    axes![Ch = 2, C = 2, E = 4, A = 2, B = 3];

    #[test]
    fn per_cluster_indexes_each_location_by_its_own_value() {
        use crate::runtime::Topology;
        type Values = SpmTensor<i32, m![Ch], m![C], m![1], m![E], Cpu>;
        let mut values = Values::new(Tensor::from_vec([
            1, 10, 20, 30, 2, 40, 50, 60, 3, 70, 80, 90, 0, 100, 110, 120,
        ]));
        let mut device = crate::Device::<Cpu>::new(Topology { chips: 2, pes: 8 }).unwrap();

        device.per_cluster(&mut values, |_, mut values| {
            let index = values.get(0u32) as usize;
            values.set(index, values.get(index) + 7);
        });

        assert_eq!(
            values.inner.into_vec(),
            [1, 17, 20, 30, 2, 40, 57, 60, 3, 70, 80, 97, 7, 100, 110, 120]
        );
    }

    /// The index type is the author's, independent of the payload's.
    #[test]
    fn per_cluster_accepts_each_integer_index_width() {
        use crate::runtime::Topology;
        type Values = SpmTensor<i32, m![Ch], m![C], m![1], m![E], Cpu>;
        let mut values = Values::zero();
        let mut device = crate::Device::<Cpu>::new(Topology { chips: 2, pes: 8 }).unwrap();

        device.per_cluster(&mut values, |_, mut values| {
            values.set(0i32, 11);
            values.set(1u32, 22);
            values.set(2i64, 33);
            values.set(3u64, 44);
            let index = values.get(0usize) as u32 - 10;
            values.set(index, values.get(index) + 5);
        });

        assert_eq!(values.inner.into_vec(), [11, 27, 33, 44].repeat(Ch::SIZE * C::SIZE));
    }

    /// The scope visits every device chip and cluster, including argument padding.
    #[test]
    fn per_cluster_visits_every_location() {
        use crate::runtime::Topology;
        type Values = SpmTensor<i32, m![1 # 2], m![1 # 2], m![1], m![E], Cpu>;
        let mut values = Values::zero();
        let mut device = crate::Device::<Cpu>::new(Topology { chips: 2, pes: 8 }).unwrap();

        device.per_cluster(&mut values, |_, mut values| values.set(0, 1));

        assert_eq!(values.inner.into_vec(), [1, 0, 0, 0].repeat(4));
    }

    #[test]
    fn an_index_addresses_a_position_and_leaves_other_pes_alone() {
        // `m![A, B # 4]` lays out two blocks of three live positions each padded to four, so
        // position 5 is the second block's middle slot, not the fifth live element.
        type Values = SpmTensor<i32, m![Ch], m![C], m![1 # 4], m![A, B # 4], Cpu>;
        const PAYLOAD: usize = 8;
        // PE 0's payload, then the three PEs the mapping leaves dead.
        const LOCATION: usize = 4 * PAYLOAD;
        const PAD: i32 = -1;
        const DEAD: i32 = -99;
        let locations = Ch::SIZE * C::SIZE;
        let payload = |location: usize| {
            let v = (location * 10) as i32;
            [v, v + 1, v + 2, PAD, v + 3, v + 4, v + 5, PAD]
        };

        let mut slots = vec![DEAD; locations * LOCATION];
        for location in 0..locations {
            slots[location * LOCATION..][..PAYLOAD].copy_from_slice(&payload(location));
        }
        let mut values = Values::new(Tensor::from_vec(slots));
        let mut device = crate::Device::<Cpu>::new(crate::runtime::Topology { chips: 2, pes: 8 }).unwrap();

        device.per_cluster(&mut values, |_, mut values| {
            let read = values.get(1usize);
            values.set(5usize, read + 100);
        });

        let actual = values.inner.into_vec();
        for location in 0..locations {
            let mut expected = payload(location);
            expected[5] = expected[1] + 100;
            assert_eq!(&actual[location * LOCATION..][..PAYLOAD], &expected);
            assert!(
                actual[location * LOCATION + PAYLOAD..][..LOCATION - PAYLOAD]
                    .iter()
                    .all(|&slot| slot == DEAD)
            );
        }
    }
}

type Placement<Chip, Cluster, Pe, Element> = Pair<Chip, Pair<Cluster, Pair<Pe, Element>>>;

/// A borrowed SPM region for DMA reads.
#[primitive(SpmTensorView)]
#[derive(Debug)]
pub struct SpmTensorView<'a, D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend = CurrentBackend> {
    pub(super) inner: TensorView<'a, D, Placement<Chip, Cluster, Pe, Element>, B>,
}

/// An exclusively borrowed SPM region for DMA writes.
#[primitive(SpmTensorViewMut)]
#[derive(Debug)]
pub struct SpmTensorViewMut<'a, D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend = CurrentBackend> {
    pub(super) inner: TensorViewMut<'a, D, Placement<Chip, Cluster, Pe, Element>, B>,
}

impl<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend> SpmTensor<D, Chip, Cluster, Pe, Element, B> {
    /// Borrows this tensor as a DMA source.
    #[primitive(SpmTensor::view)]
    pub fn view(&self) -> SpmTensorView<'_, D, Chip, Cluster, Pe, Element, B> {
        SpmTensorView {
            inner: self.inner.view(),
        }
    }

    /// Borrows this tensor as a DMA destination.
    #[primitive(SpmTensor::view_mut)]
    pub fn view_mut(&mut self) -> SpmTensorViewMut<'_, D, Chip, Cluster, Pe, Element, B> {
        SpmTensorViewMut {
            inner: self.inner.view_mut(),
        }
    }
}

impl<'a, D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend>
    SpmTensorView<'a, D, Chip, Cluster, Pe, Element, B>
{
    /// Selects an Element window for DMA reads using a compile-time start offset.
    #[primitive(SpmTensorView::tile)]
    pub fn tile<I: M, const LEN: usize, E2: M>(
        self,
        start: impl RuntimeIndex,
    ) -> SpmTensorView<'a, D, Chip, Cluster, Pe, E2, B> {
        let start = start.into_usize();
        assert_tile::<I, Element, E2, LEN>(PaddingKind::Top);
        SpmTensorView {
            inner: self.inner.retile::<I, _>(start, LEN),
        }
    }

    /// Selects a PE window for DMA reads using a compile-time start offset.
    #[primitive(SpmTensorView::pe_tile)]
    pub fn pe_tile<I: M, const LEN: usize, P2: M>(
        self,
        start: impl RuntimeIndex,
    ) -> SpmTensorView<'a, D, Chip, Cluster, P2, Element, B> {
        let start = start.into_usize();
        assert_tile::<I, Pe, P2, LEN>(PaddingKind::Top);
        SpmTensorView {
            inner: self.inner.retile::<I, _>(start, LEN),
        }
    }
}

impl<'a, D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend>
    SpmTensorViewMut<'a, D, Chip, Cluster, Pe, Element, B>
{
    /// Selects an Element window for DMA writes using a compile-time start offset.
    #[primitive(SpmTensorViewMut::tile)]
    pub fn tile<I: M, const LEN: usize, E2: M>(
        self,
        start: impl RuntimeIndex,
    ) -> SpmTensorViewMut<'a, D, Chip, Cluster, Pe, E2, B> {
        let start = start.into_usize();
        assert_tile::<I, Element, E2, LEN>(PaddingKind::Bottom);
        SpmTensorViewMut {
            inner: self.inner.retile::<I, _>(start, LEN),
        }
    }

    /// Selects a PE window for DMA writes using a compile-time start offset.
    #[primitive(SpmTensorViewMut::pe_tile)]
    pub fn pe_tile<I: M, const LEN: usize, P2: M>(
        self,
        start: impl RuntimeIndex,
    ) -> SpmTensorViewMut<'a, D, Chip, Cluster, P2, Element, B> {
        let start = start.into_usize();
        assert_tile::<I, Pe, P2, LEN>(PaddingKind::Bottom);
        SpmTensorViewMut {
            inner: self.inner.retile::<I, _>(start, LEN),
        }
    }
}
