//! Builds deferred HBM scatters by declaring the indexed axis, payload, and index domain.

use super::gather::{Begin, IndexSource};
use super::*;
use crate::backend::indirect::IndexUnit;
use furiosa_opt_lower::{
    DmPlacement, IndexMapping, ScatterPayload, SpmPlacement, scatter_payload, validate_index_chip,
    validate_scatter_payload, validate_scatter_updates, validate_spm_indirect_index,
};
use std::num::NonZeroUsize;

/// A deferred scatter whose `Stage` determines whether an index is available.
#[primitive(ScatterPlan)]
#[derive(Debug)]
pub struct ScatterPlan<
    'd,
    D: Scalar,
    Chip: M,
    DstElement: M,
    IndexedAxis: M,
    Payload: M,
    Stage,
    B: Backend = CurrentBackend,
> where
    Stage: ScatterStage<B>,
{
    /// Written by device DMA alone, so it need not span every chip.
    destination: HbmTensorViewMut<'d, D, Chip, DstElement, B, SomeChips>,
    /// The destination payload and indexed-axis stride.
    payload: ScatterPayload,
    index: Stage::IndexTensor<'d>,
    _marker: PhantomData<(IndexedAxis, Payload)>,
}

/// Values available at each scatter stage.
#[doc(hidden)]
pub trait ScatterStage<B: Backend> {
    /// The index tensor this stage reads for `'l`, or `()` before one is supplied.
    type IndexTensor<'l>;
}

/// A stage with an index, ready to run over `Domain`.
#[doc(hidden)]
pub trait IndexedScatter<B: Backend>: ScatterStage<B> {
    /// The index domain, which determines how updates are grouped.
    type Domain: M;
}

/// SPM positions over `Domain`, the named cluster axes followed by the index list.
#[doc(hidden)]
#[derive(Debug)]
pub struct WithPositions<IdxAxes, Domain>(PhantomData<(IdxAxes, Domain)>);

/// Byte offsets held in HBM; the index's own mapping is the domain.
#[doc(hidden)]
#[derive(Debug)]
pub struct WithByteOffsets<IdxChip, Domain>(PhantomData<(IdxChip, Domain)>);

impl<B: Backend> ScatterStage<B> for Begin {
    type IndexTensor<'l> = ();
}

impl<IdxAxes: M, Domain: M, B: Backend> ScatterStage<B> for WithPositions<IdxAxes, Domain> {
    type IndexTensor<'l> = ScatterIndexTensor<Tensor<i32, IdxAxes, B>>;
}

impl<IdxChip: M, Domain: M, B: Backend> ScatterStage<B> for WithByteOffsets<IdxChip, Domain> {
    type IndexTensor<'l> = ScatterIndexTensor<HbmTensorView<'l, i32, IdxChip, Domain, B>>;
}

impl<IdxAxes: M, Domain: M, B: Backend> IndexedScatter<B> for WithPositions<IdxAxes, Domain> {
    type Domain = Domain;
}

impl<IdxChip: M, Domain: M, B: Backend> IndexedScatter<B> for WithByteOffsets<IdxChip, Domain> {
    type Domain = Domain;
}

/// The index tensor carried by a scatter plan, with the unit its values count in.
#[derive(Debug)]
pub struct ScatterIndexTensor<Source> {
    tensor: Source,
    unit: IndexUnit,
    /// The index domain, checked against the declaration.
    domain: Mapping,
}

/// Returns the index domain, checking it against `Domain`.
fn declared_domain<Domain: M>(index: IndexMapping<'_>) -> Mapping {
    let domain = index.domain().unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        domain,
        Domain::to_value().normalize(),
        "scatter domain must be the one the index carries"
    );
    domain
}

impl<'d, D: Scalar, Chip: M, DstElement: M, B: Backend, Chips> HbmTensorViewMut<'d, D, Chip, DstElement, B, Chips> {
    /// Starts a scatter through `IndexedAxis`, writing `Payload` per index entry.
    /// `Payload` describes one update entry, including padding.
    #[primitive(HbmTensorViewMut::scatter)]
    pub fn scatter<IndexedAxis: M, Payload: M>(
        self,
    ) -> ScatterPlan<'d, D, Chip, DstElement, IndexedAxis, Payload, Begin, B> {
        let payload = scatter_payload(&DstElement::to_value(), &IndexedAxis::to_value())
            .unwrap_or_else(|error| panic!("{error}"));
        validate_scatter_payload(&Payload::to_value(), &payload).unwrap_or_else(|error| panic!("{error}"));
        ScatterPlan {
            destination: HbmTensorViewMut {
                inner: self.inner,
                chips: PhantomData,
            },
            payload,
            index: (),
            _marker: PhantomData,
        }
    }
}

impl<'d, D: Scalar, Chip: M, DstElement: M, IndexedAxis: M, Payload: M, Domain: M, Source, Stage, B: Backend>
    ScatterPlan<'d, D, Chip, DstElement, IndexedAxis, Payload, Stage, B>
where
    Stage: IndexedScatter<B, Domain = Domain, IndexTensor<'d> = ScatterIndexTensor<Source>>,
    Source: IndexSource<B>,
{
    /// Runs the scatter, consuming the DM tensor it takes the values from.
    #[primitive(ScatterPlan::from_dm)]
    pub fn from_dm<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        updates: DmTensor<D, Chip, Cluster, Slice, Element, B>,
    ) where
        D: MaterializableScalar,
    {
        self.from_dm_view(dma, updates.view());
    }

    /// Runs the scatter from a DM view, checking that it supplies `Payload` once per domain entry.
    #[primitive(ScatterPlan::from_dm_view)]
    pub fn from_dm_view<'u, Cluster: M, Slice: M, Element: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
        updates: DmTensorView<'u, D, Chip, Cluster, Slice, Element, B>,
    ) where
        D: MaterializableScalar,
    {
        let domain = self.index.domain.clone();
        let declared_payload = Payload::to_value();
        validate_scatter_updates(
            DmPlacement {
                cluster: &Cluster::to_value(),
                slice: &Slice::to_value(),
                element: &Element::to_value(),
            },
            &declared_payload,
            &domain,
        )
        .unwrap_or_else(|error| panic!("{error}"));

        let updates = updates.inner.read();
        let index = self.index.tensor.read(dma);
        self.destination.write(dma, |mut destination| {
            let mut output = destination.read();
            updates.scatter::<Domain, IndexedAxis, _, _>(&mut output, &index, &Chip::to_value(), self.index.unit);
            destination.transpose(output.view(), false);
        });
    }
}

/// Adds SPM positions to a scatter, inferring placement from the index tensor.
#[doc(hidden)]
pub trait ScatterPositions<
    'd,
    'i,
    D: Scalar,
    Chip: M,
    DstElement: M,
    IndexedAxis: M,
    Payload: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    B: Backend,
>
{
    /// Supplies unscaled positions over `Domain`: the named cluster axes followed by the list.
    /// Broadcast or padded single-copy clusters share one list and do not join `Domain`.
    #[furiosa_opt::primitive = "ScatterPlan::by_positions"]
    fn by_positions<Domain: M>(
        self,
        indices: &'i SpmTensor<i32, IdxChip, IdxCluster, IdxPe, IdxElement, B>,
    ) -> ScatterPlan<
        'd,
        D,
        Chip,
        DstElement,
        IndexedAxis,
        Payload,
        WithPositions<Pair<IdxChip, Pair<IdxCluster, Pair<IdxPe, IdxElement>>>, Domain>,
        B,
    >;
}

impl<
    'd,
    'i,
    D: Scalar,
    Chip: M,
    DstElement: M,
    IndexedAxis: M,
    Payload: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    B: Backend,
> ScatterPositions<'d, 'i, D, Chip, DstElement, IndexedAxis, Payload, IdxChip, IdxCluster, IdxPe, IdxElement, B>
    for ScatterPlan<'d, D, Chip, DstElement, IndexedAxis, Payload, Begin, B>
{
    #[furiosa_opt::primitive = "ScatterPlan::by_positions"]
    fn by_positions<Domain: M>(
        self,
        indices: &'i SpmTensor<i32, IdxChip, IdxCluster, IdxPe, IdxElement, B>,
    ) -> ScatterPlan<
        'd,
        D,
        Chip,
        DstElement,
        IndexedAxis,
        Payload,
        WithPositions<Pair<IdxChip, Pair<IdxCluster, Pair<IdxPe, IdxElement>>>, Domain>,
        B,
    > {
        validate_index_chip(&IdxChip::to_value(), &Chip::to_value()).unwrap_or_else(|error| panic!("{error}"));
        validate_spm_indirect_index(&IdxPe::to_value()).unwrap_or_else(|error| panic!("{error}"));
        let (cluster, pe, element) = (IdxCluster::to_value(), IdxPe::to_value(), IdxElement::to_value());
        let domain = declared_domain::<Domain>(IndexMapping::SpmPositions(SpmPlacement {
            cluster: &cluster,
            pe: &pe,
            element: &element,
        }));
        ScatterPlan {
            destination: self.destination,
            payload: self.payload,
            index: ScatterIndexTensor {
                tensor: indices.inner.clone(),
                unit: IndexUnit::Positions,
                domain,
            },
            _marker: PhantomData,
        }
    }
}

/// Adds byte offsets to a scatter, inferring the index tensor's chip mapping.
#[doc(hidden)]
pub trait ScatterByteOffsets<
    'd,
    'i: 'd,
    D: Scalar,
    Chip: M,
    DstElement: M,
    IndexedAxis: M,
    Payload: M,
    IdxChip: M,
    B: Backend,
>
{
    /// Supplies byte offsets into the destination, with `Domain` inferred from the index mapping.
    /// The index chip mapping must match the destination's or broadcast across it.
    #[furiosa_opt::primitive = "ScatterPlan::by_byte_offsets"]
    fn by_byte_offsets<Domain: M>(
        self,
        offsets: HbmTensorView<'i, i32, IdxChip, Domain, B>,
    ) -> ScatterPlan<'d, D, Chip, DstElement, IndexedAxis, Payload, WithByteOffsets<IdxChip, Domain>, B>;
}

impl<'d, 'i: 'd, D: Scalar, Chip: M, DstElement: M, IndexedAxis: M, Payload: M, IdxChip: M, B: Backend>
    ScatterByteOffsets<'d, 'i, D, Chip, DstElement, IndexedAxis, Payload, IdxChip, B>
    for ScatterPlan<'d, D, Chip, DstElement, IndexedAxis, Payload, Begin, B>
{
    #[furiosa_opt::primitive = "ScatterPlan::by_byte_offsets"]
    fn by_byte_offsets<Domain: M>(
        self,
        offsets: HbmTensorView<'i, i32, IdxChip, Domain, B>,
    ) -> ScatterPlan<'d, D, Chip, DstElement, IndexedAxis, Payload, WithByteOffsets<IdxChip, Domain>, B> {
        validate_index_chip(&IdxChip::to_value(), &Chip::to_value()).unwrap_or_else(|error| panic!("{error}"));
        let unit = IndexUnit::Bytes(indexed_axis_stride::<D>(&self.payload));
        let domain = declared_domain::<Domain>(IndexMapping::ByteOffsets(&Domain::to_value()));
        ScatterPlan {
            destination: self.destination,
            payload: self.payload,
            index: ScatterIndexTensor {
                tensor: offsets,
                unit,
                domain,
            },
            _marker: PhantomData,
        }
    }
}

/// The byte stride between indexed-axis positions in the destination.
fn indexed_axis_stride<D: Scalar>(payload: &ScatterPayload) -> NonZeroUsize {
    let stride = payload
        .indexed_axis_stride_bytes(D::BITS)
        .unwrap_or_else(|error| panic!("{error}"));
    NonZeroUsize::new(stride).expect("a destination stride spans at least one byte")
}
