//! Builds deferred HBM gathers by declaring the payload, index domain, and output placement.
//!
//! A `GatherPlan` carries its stage in a type parameter, so the type checker enforces the order
//! the steps run in. Each block below lists every outgoing edge from that stage. The set is
//! normative: it must equal the `GatherPlan<.., Stage, ..>` impl targets below.
//!
//! # Stage graph
//!
//! ```text
//! HbmTensor
//!     └── gather::<IndexedAxis, Payload>  →  Begin
//!
//! Begin
//!     ├── by_byte_offsets::<Domain>       →  IndexedByteOffsets   (index in HBM)
//!     └── by_positions::<Domain>          →  IndexedPositions     (index in SPM)
//!
//! IndexedByteOffsets
//!     ├── sparse_prefix                   →  Prefixed<IndexedByteOffsets>
//!     └── to_dm                           →  DmTensor
//!
//! IndexedPositions
//!     └── to_dm                           →  DmTensor
//!
//! Prefixed<IndexedByteOffsets>
//!     └── to_dm                           →  DmTensor
//! ```
//!
//! An index is supplied once: `Begin` is the only stage the two `by_*` steps apply to, and nothing
//! returns to it. `sparse_prefix` takes a runtime length, and only the byte-offset stage carries
//! it, so an SPM index always gathers its whole domain. Its `to_dm` is written over any indexed
//! stage, but `Prefixed<IndexedPositions>` has no producer.
//!
//! Every `to_dm` writes DM, so it checks the destination tail against the write granularity, which
//! is [`super::layout`]'s rule.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::context::*;
use crate::scalar::*;
use crate::tensor::*;

use super::layout::check_dma_tail;
use crate::backend::indirect::IndexUnit;
use crate::tensor::memory::*;
use furiosa_opt_lower::{
    DM_WRITE_ALIGN_BYTES, DmPlacement, GatherPayload, IndexMapping, SpmPlacement, validate_gather_output,
    validate_gather_payload, validate_index_chip, validate_spm_indirect_index,
};
use std::num::NonZeroUsize;

/// A deferred gather whose `Stage` determines the available index and prefix.
#[primitive(GatherPlan)]
#[derive(Debug)]
pub struct GatherPlan<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, Stage, B: Backend = CurrentBackend>
where
    Stage: GatherStage<B>,
{
    table: &'l HbmTensor<D, Chip, TableAxes, B>,
    /// The checked payload and indexed-axis stride.
    payload: GatherPayload,
    index: Stage::IndexTensor<'l>,
    prefix: Stage::Prefix,
    _marker: PhantomData<IndexedAxis>,
}

/// Values available at each gather stage.
#[doc(hidden)]
pub trait GatherStage<B: Backend>: sealed::Sealed {
    /// The index tensor this stage reads for `'l`, or `()` before one is supplied.
    type IndexTensor<'l>;
    /// The runtime prefix that shortens it, or `()` when every position runs.
    type Prefix;
}

/// A stage with an index, ready for an output placement.
#[doc(hidden)]
pub trait IndexedStage<B: Backend>: GatherStage<B> {}

mod sealed {
    /// Seals the gather stages to this crate.
    pub trait Sealed {}
}

/// The payload is settled and no index has been supplied.
#[primitive(Begin)]
#[doc(hidden)]
#[derive(Debug)]
pub struct Begin;

/// Byte offsets held in HBM have been supplied; the index's own mapping is the domain.
#[primitive(IndexedByteOffsets)]
#[doc(hidden)]
#[derive(Debug)]
pub struct IndexedByteOffsets<Chip: M, Domain: M>(PhantomData<(Chip, Domain)>);

/// SPM positions over `Domain`, the named cluster axes followed by the index list.
#[primitive(IndexedPositions)]
#[doc(hidden)]
#[derive(Debug)]
pub struct IndexedPositions<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M>(
    PhantomData<(Chip, IdxCluster, IdxPe, IdxElement, Domain)>,
);

/// Runs a runtime-valid prefix of the index positions.
#[primitive(Prefixed)]
#[doc(hidden)]
#[derive(Debug)]
pub struct Prefixed<Stage>(PhantomData<Stage>);

impl sealed::Sealed for Begin {}
impl<Chip: M, Domain: M> sealed::Sealed for IndexedByteOffsets<Chip, Domain> {}
impl<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M> sealed::Sealed
    for IndexedPositions<Chip, IdxCluster, IdxPe, IdxElement, Domain>
{
}
impl<Stage> sealed::Sealed for Prefixed<Stage> {}

impl<B: Backend> GatherStage<B> for Begin {
    type IndexTensor<'l> = ();
    type Prefix = ();
}

impl<Chip: M, Domain: M, B: Backend> GatherStage<B> for IndexedByteOffsets<Chip, Domain> {
    type IndexTensor<'l> = GatherIndexTensor<HbmTensorView<'l, i32, Chip, Domain, B>>;
    type Prefix = ();
}

impl<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M, B: Backend> GatherStage<B>
    for IndexedPositions<Chip, IdxCluster, IdxPe, IdxElement, Domain>
{
    type IndexTensor<'l> = GatherIndexTensor<Tensor<i32, Pair<Chip, Pair<IdxCluster, Pair<IdxPe, IdxElement>>>, B>>;
    type Prefix = ();
}

impl<Stage: GatherStage<B>, B: Backend> GatherStage<B> for Prefixed<Stage> {
    type IndexTensor<'l> = Stage::IndexTensor<'l>;
    type Prefix = RuntimePrefix;
}

impl<Chip: M, Domain: M, B: Backend> IndexedStage<B> for IndexedByteOffsets<Chip, Domain> {}
impl<Chip: M, IdxCluster: M, IdxPe: M, IdxElement: M, Domain: M, B: Backend> IndexedStage<B>
    for IndexedPositions<Chip, IdxCluster, IdxPe, IdxElement, Domain>
{
}

/// A gather index with its entry domain and value unit.
#[derive(Debug)]
pub struct GatherIndexTensor<Source> {
    tensor: Source,
    domain: Mapping,
    unit: IndexUnit,
}

/// An index tensor that is read when the DMA using it executes.
#[doc(hidden)]
pub trait IndexSource<B: Backend> {
    /// The mapping of the index values.
    type Axes: M;

    /// Reads the index, from HBM when it lives there.
    fn read(&self, dma: &DmaContext<{ Dma::Tensor }, B>) -> Tensor<i32, Self::Axes, B>;
}

impl<Axes: M, B: Backend> IndexSource<B> for Tensor<i32, Axes, B> {
    type Axes = Axes;

    fn read(&self, _: &DmaContext<{ Dma::Tensor }, B>) -> Tensor<i32, Axes, B> {
        self.clone()
    }
}

impl<Chip: M, Element: M, B: Backend> IndexSource<B> for HbmTensorView<'_, i32, Chip, Element, B> {
    type Axes = Pair<Chip, Element>;

    fn read(&self, dma: &DmaContext<{ Dma::Tensor }, B>) -> Tensor<i32, Pair<Chip, Element>, B> {
        HbmTensorView::read(self, dma, |view| view.read())
    }
}

/// The runtime length requested for a sparse gather.
#[derive(Debug)]
pub struct RuntimePrefix {
    valid_length: i32,
}

impl<D: Scalar, Chip: M, TableAxes: M, B: Backend> HbmTensor<D, Chip, TableAxes, B> {
    /// Starts a gather that replaces `IndexedAxis`, reading `Payload` per index entry.
    /// `Payload` includes any padding left after removing the indexed axis.
    #[primitive(HbmTensor::gather)]
    pub fn gather<IndexedAxis: M, Payload: M>(&self) -> GatherPlan<'_, D, Chip, TableAxes, IndexedAxis, Begin, B> {
        let payload = validate_gather_payload(&TableAxes::to_value(), &IndexedAxis::to_value(), &Payload::to_value())
            .unwrap_or_else(|error| panic!("{error}"));
        GatherPlan {
            table: self,
            payload,
            index: (),
            prefix: (),
            _marker: PhantomData,
        }
    }
}

/// Adds a byte-offset index to a gather plan.
#[doc(hidden)]
pub trait GatherByteOffsets<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, B: Backend> {
    /// Supplies byte offsets into the table, with `Domain` inferred from the index mapping.
    ///
    /// The index chip axis must match the table's chip axis or broadcast across it.
    #[furiosa_opt::primitive = "GatherPlan::by_byte_offsets"]
    fn by_byte_offsets<Domain: M>(
        self,
        index: &'l HbmTensor<i32, IdxChip, Domain, B>,
    ) -> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedByteOffsets<IdxChip, Domain>, B>;
}

impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, B: Backend>
    GatherByteOffsets<'l, D, Chip, TableAxes, IndexedAxis, IdxChip, B>
    for GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Begin, B>
{
    #[furiosa_opt::primitive = "GatherPlan::by_byte_offsets"]
    fn by_byte_offsets<Domain: M>(
        self,
        index: &'l HbmTensor<i32, IdxChip, Domain, B>,
    ) -> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedByteOffsets<IdxChip, Domain>, B> {
        validate_index_chip(&IdxChip::to_value(), &Chip::to_value()).unwrap_or_else(|error| panic!("{error}"));
        let unit = IndexUnit::Bytes(indexed_axis_stride::<D>(&self.payload));
        let domain = IndexMapping::ByteOffsets(&Domain::to_value())
            .domain()
            .unwrap_or_else(|error| panic!("{error}"));
        GatherPlan {
            table: self.table,
            payload: self.payload,
            index: GatherIndexTensor {
                tensor: index.view(),
                domain,
                unit,
            },
            prefix: (),
            _marker: PhantomData,
        }
    }
}

/// Adds SPM positions to a gather, inferring placement from the index tensor.
#[doc(hidden)]
pub trait GatherPositions<
    'l,
    D: Scalar,
    Chip: M,
    TableAxes: M,
    IndexedAxis: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    B: Backend,
>
{
    /// Supplies unscaled positions over `Domain`: the named cluster axes followed by the list.
    /// Broadcast or padded single-copy clusters share one list and do not join `Domain`.
    #[furiosa_opt::primitive = "GatherPlan::by_positions"]
    fn by_positions<Domain: M>(
        self,
        index: &'l SpmTensor<i32, IdxChip, IdxCluster, IdxPe, IdxElement, B>,
    ) -> GatherPlan<
        'l,
        D,
        Chip,
        TableAxes,
        IndexedAxis,
        IndexedPositions<IdxChip, IdxCluster, IdxPe, IdxElement, Domain>,
        B,
    >;
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    TableAxes: M,
    IndexedAxis: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    B: Backend,
> GatherPositions<'l, D, Chip, TableAxes, IndexedAxis, IdxChip, IdxCluster, IdxPe, IdxElement, B>
    for GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Begin, B>
{
    #[furiosa_opt::primitive = "GatherPlan::by_positions"]
    fn by_positions<Domain: M>(
        self,
        index: &'l SpmTensor<i32, IdxChip, IdxCluster, IdxPe, IdxElement, B>,
    ) -> GatherPlan<
        'l,
        D,
        Chip,
        TableAxes,
        IndexedAxis,
        IndexedPositions<IdxChip, IdxCluster, IdxPe, IdxElement, Domain>,
        B,
    > {
        validate_index_chip(&IdxChip::to_value(), &Chip::to_value()).unwrap_or_else(|error| panic!("{error}"));
        validate_spm_indirect_index(&IdxPe::to_value()).unwrap_or_else(|error| panic!("{error}"));
        let (cluster, pe, element) = (IdxCluster::to_value(), IdxPe::to_value(), IdxElement::to_value());
        let index_mapping = IndexMapping::SpmPositions(SpmPlacement {
            cluster: &cluster,
            pe: &pe,
            element: &element,
        });
        let domain = index_mapping.domain().unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            domain,
            Domain::to_value().normalize(),
            "gather domain must be the one the index carries"
        );
        GatherPlan {
            table: self.table,
            payload: self.payload,
            index: GatherIndexTensor {
                tensor: index.inner.clone(),
                domain,
                unit: IndexUnit::Positions,
            },
            prefix: (),
            _marker: PhantomData,
        }
    }
}

impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, Source: IndexSource<B>, Stage, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Stage, B>
where
    Stage: IndexedStage<B, Prefix = (), IndexTensor<'l> = GatherIndexTensor<Source>>,
{
    fn materialize<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &DmaContext<{ Dma::Tensor }, B>,
        prefix: Option<usize>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        validate_placement::<Cluster, Slice, Element>(&self.payload, &self.index.domain);
        // The source element is the payload one index entry selects. Only the tail rule applies: the
        // stream strides are `validate_placement`'s, just above.
        check_dma_tail::<D>(&self.payload.payload, &Element::to_value(), DM_WRITE_ALIGN_BYTES);
        let mut output = DmTensor::from_parts(Tensor::zeroed());
        self.table.read(dma).gather::<IndexedAxis, _, _>(
            &mut output.inner,
            &self.index.tensor.read(dma),
            &self.index.domain,
            &Chip::to_value(),
            prefix,
            self.index.unit,
        );
        output
    }
}

// Separate inherent impls avoid overlap with prefixed plans; primitive attributes cannot be shared
// through `macro_rules!`.
impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, IdxChip: M, Domain: M, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedByteOffsets<IdxChip, Domain>, B>
{
    /// Materializes the gather in the requested data-memory shape.
    #[primitive(GatherPlan::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        self.materialize(dma, None)
    }

    /// Limits each chip's gathered entries to `valid_length`, in `Domain` order; zero or less writes none.
    /// All indices must remain valid: DMA may execute past the prefix, and the CPU validates the whole list.
    #[primitive(GatherPlan::sparse_prefix)]
    pub fn sparse_prefix(
        self,
        valid_length: i32,
    ) -> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Prefixed<IndexedByteOffsets<IdxChip, Domain>>, B> {
        GatherPlan {
            table: self.table,
            payload: self.payload,
            index: self.index,
            prefix: RuntimePrefix { valid_length },
            _marker: PhantomData,
        }
    }
}

impl<
    'l,
    D: Scalar,
    Chip: M,
    TableAxes: M,
    IndexedAxis: M,
    IdxChip: M,
    IdxCluster: M,
    IdxPe: M,
    IdxElement: M,
    Domain: M,
    B: Backend,
> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, IndexedPositions<IdxChip, IdxCluster, IdxPe, IdxElement, Domain>, B>
{
    /// Materializes the gather in the requested data-memory shape.
    #[primitive(GatherPlan::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        self.materialize(dma, None)
    }
}

impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, Source: IndexSource<B>, Stage, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Prefixed<Stage>, B>
where
    Stage: IndexedStage<B, Prefix = (), IndexTensor<'l> = GatherIndexTensor<Source>>,
{
    /// Materializes the gather in the requested data-memory shape.
    #[primitive(GatherPlan::to_dm)]
    pub fn to_dm<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &mut DmaContext<{ Dma::Tensor }, B>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        let RuntimePrefix { valid_length } = self.prefix;
        let requested = usize::try_from(valid_length).unwrap_or(0);
        let plan = GatherPlan::<'l, D, Chip, TableAxes, IndexedAxis, Stage, B> {
            table: self.table,
            payload: self.payload,
            index: self.index,
            prefix: (),
            _marker: PhantomData,
        };
        plan.materialize(dma, Some(requested))
    }
}

fn indexed_axis_stride<D: Scalar>(payload: &GatherPayload) -> NonZeroUsize {
    let stride = payload
        .indexed_axis_stride_bytes(D::BITS)
        .unwrap_or_else(|error| panic!("{error}"));
    NonZeroUsize::new(stride).expect("an indexed-axis stride spans at least one byte")
}

fn validate_placement<Cluster: M, Slice: M, Element: M>(payload: &GatherPayload, domain: &Mapping) {
    let (cluster, slice, element) = (Cluster::to_value(), Slice::to_value(), Element::to_value());
    validate_gather_output(
        &payload.payload,
        domain,
        DmPlacement {
            cluster: &cluster,
            slice: &slice,
            element: &element,
        },
    )
    .unwrap_or_else(|error| panic!("{error}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::{Allocation, Cpu};
    use crate::constraints;
    use crate::runtime::{Device, Topology};

    /// Runs an SPM-index gather and returns its values with a host oracle.
    fn run_gather<B: Backend>() -> (Vec<i32>, Vec<i32>) {
        axes![W = 8, V = 2, K = 128];
        type Slice = m![K / 2];
        // Padding-only PE axis spanning the PEs those slices cover: one PE here, 4 on a full cluster.
        type DummyPe = m![1 # { Slice::SIZE / constraints::SLICES_PER_PE }];
        // Non-monotonic and includes both table boundaries, exposing ignored indices.
        const PERM: [i32; 8] = [3, 7, 1, 5, 0, 6, 2, 4];
        let row = |k: usize| PERM[k % W::SIZE];

        let table_buf: Vec<i32> = (0..W::SIZE as i32).flat_map(|r| [10 * r, 10 * r + 1]).collect();
        let idx_buf: Vec<i32> = (0..K::SIZE).map(row).collect();
        let expected: Vec<i32> = (0..K::SIZE).flat_map(|k| [10 * row(k), 10 * row(k) + 1]).collect();

        let index_dm = DmTensor::<i32, m![1], m![1], Slice, m![K % 2], B>::from_parts(Tensor::from_vec(idx_buf));
        let mut tdma = DmaContext::<{ Dma::Tensor }, B>::on(
            B::open(vec![crate::Placement {
                chips: vec![0],
                pes: 0..4,
            }])
            .unwrap(),
        );
        let table = HbmTensor::<i32, m![1], m![W, V], B>::from_value(&tdma, &Tensor::from_vec(table_buf));
        let index: SpmTensor<i32, m![1], m![1], DummyPe, m![K], B> = index_dm.to_spm(&mut tdma);

        let output: DmTensor<i32, m![1], m![1], Slice, m![K % 2, V], B> = table
            .gather::<m![W], m![V]>()
            .by_positions::<m![K]>(&index)
            .to_dm::<m![1], Slice, m![K % 2, V]>(&mut tdma);
        (output.inner.into_vec(), expected)
    }

    #[test]
    fn cpu_gather_roundtrip() {
        let (got, expected) = run_gather::<Cpu>();
        assert_eq!(got, expected);
    }

    /// A deferred gather reads aliased HBM indices when its DMA executes.
    #[test]
    fn gathers_current_indices() {
        axes![W = 2, V = 2, K = 128];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![W, V], Cpu>::from_value(&device.tdma, &Tensor::from_vec([0, 1, 10, 11]));
        let index = HbmTensor::<i32, m![1], m![K], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec(std::iter::repeat_n(0, K::SIZE)),
        );
        let mut alias =
            HbmTensor::<i32, m![1], m![K], Cpu>::try_from(crate::HbmBuffer(index.buffer.alias(..).unwrap())).unwrap();
        let plan = table.gather::<m![W], m![V]>().by_byte_offsets::<m![K]>(&index);
        alias.store(&device.tdma, &Tensor::from_vec(std::iter::repeat_n(8, K::SIZE)));

        let output = plan.to_dm::<m![1], m![K / 2], m![K % 2, V]>(&mut device.tdma);

        assert_eq!(output.inner.into_vec(), [10, 11].repeat(K::SIZE));
    }
}
