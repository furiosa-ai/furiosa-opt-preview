//! Builds deferred HBM gathers by declaring the payload, index domain, and output placement.

use super::*;
use crate::backend::indirect::IndexUnit;
use furiosa_opt_lower::{
    DmPlacement, GatherPayload, IndexMapping, SpmPlacement, validate_gather_output, validate_gather_payload,
    validate_index_chip, validate_spm_indirect_index,
};
use std::num::NonZeroUsize;

mod destination;
mod unscaled;
mod window;

/// A deferred gather whose `Stage` determines the available index and prefix.
#[primitive(GatherPlan)]
#[derive(Debug)]
pub struct GatherPlan<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, Stage, B: Backend = CurrentBackend>
where
    Stage: GatherStage<B>,
{
    table: GatherTable<'l, D, Chip, TableAxes, B>,
    /// The checked payload and indexed-axis stride.
    payload: GatherPayload,
    index: Stage::IndexTensor<'l>,
    prefix: Stage::Prefix,
    _marker: PhantomData<(Chip, TableAxes, IndexedAxis)>,
}

#[derive(Debug)]
struct GatherTable<'l, D: Scalar, Chip: M, Element: M, B: Backend> {
    source: TableSource<'l, D, Chip, Element, B>,
    prefix: Option<BatchPrefix<'l, B>>,
}

#[derive(Debug)]
enum TableSource<'l, D: Scalar, Chip: M, Element: M, B: Backend> {
    /// One tensor's storage, read when the gather executes.
    Tensor(&'l HbmTensor<D, Chip, Element, B>),
    /// One view's cells, read when the gather executes.
    View(HbmTensorView<'l, D, Chip, Element, B>),
    /// Slice blocks from the selected entry onward, so indices cannot reach the blocks before it.
    Blocks(&'l [HbmTensor<D, Chip, Element, B>]),
    /// One field window of each slice block, read when the gather executes.
    Window(Vec<HbmTensorView<'l, D, Chip, Element, B>>),
}

#[derive(Debug)]
struct BatchPrefix<'l, B: Backend> {
    lengths: HbmBufferSlice<'l, B>,
    batch: Mapping,
    sparse: Mapping,
    offset: i32,
}

impl<'l, D: Scalar, Chip: M, Element: M, B: Backend> HbmTensorView<'l, D, Chip, Element, B> {
    /// Starts a staged gather of fixed payloads from a native byte allocation.
    /// Byte offsets supplied to the plan must select aligned windows within its actual capacity.
    #[primitive(HbmTensorView::gather_buffer)]
    pub fn gather_buffer(_buffer: &'l crate::HbmBuffer<B>) -> GatherPlan<'l, D, Chip, Element, Identity, Begin, B> {
        panic!("native buffer gathers require device execution");
    }
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

impl<'l, D: Scalar, Chip: M, TableAxes: M, IndexedAxis: M, Stage: GatherStage<B>, B: Backend>
    GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Stage, B>
{
    /// Addresses indices relative to the selected runtime slice entry.
    #[primitive(GatherPlan::entry)]
    pub fn entry(mut self, entry: usize) -> Self {
        match &mut self.table.source {
            TableSource::Tensor(_) | TableSource::View(_) => panic!("an entry offset requires slice storage"),
            TableSource::Blocks(blocks) => {
                assert!(entry < blocks.len(), "table entry {entry} exceeds allocated entries");
                *blocks = &blocks[entry..];
            }
            TableSource::Window(blocks) => {
                assert!(entry < blocks.len(), "table entry {entry} exceeds allocated entries");
                blocks.drain(..entry);
            }
        }
        self
    }

    /// Runs the gather without the device's check of each offset against the slice.
    ///
    /// # Safety
    /// Every offset the gather reads must address a whole payload within one block of the slice
    /// from the selected entry onward.
    #[primitive(GatherPlan::unchecked)]
    pub unsafe fn unchecked(self) -> Self {
        assert!(
            matches!(self.table.source, TableSource::Blocks(_) | TableSource::Window(_)),
            "only a slice gather checks its offsets"
        );
        self
    }
}

impl<D: Scalar, Chip: M, TableAxes: M, B: Backend> HbmTensor<D, Chip, TableAxes, B> {
    /// Starts a gather that replaces `IndexedAxis`, reading `Payload` per index entry.
    /// `Payload` includes any padding left after removing the indexed axis.
    #[primitive(HbmTensor::gather)]
    pub fn gather<IndexedAxis: M, Payload: M>(&self) -> GatherPlan<'_, D, Chip, TableAxes, IndexedAxis, Begin, B> {
        let payload = validate_gather_payload(&TableAxes::to_value(), &IndexedAxis::to_value(), &Payload::to_value())
            .unwrap_or_else(|error| panic!("{error}"));
        GatherPlan {
            table: GatherTable {
                source: TableSource::Tensor(self),
                prefix: None,
            },
            payload,
            index: (),
            prefix: (),
            _marker: PhantomData,
        }
    }
}

impl<D: MaterializableScalar, Chip: M, Element: M, B: Backend> slice::Slice<HbmTensor<D, Chip, Element, B>> {
    /// Starts a gather without changing collection membership.
    #[primitive(HbmTensor::gather)]
    pub fn gather<IndexedAxis: M, Payload: M>(&self) -> GatherPlan<'_, D, Chip, Element, IndexedAxis, Begin, B> {
        let rows = slice::outer::<Chip, Element>(2);
        let payload = validate_gather_payload(
            &rows.clone().pair(Element::to_value()),
            &rows.pair(IndexedAxis::to_value()),
            &Payload::to_value(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
        GatherPlan {
            table: GatherTable {
                source: TableSource::Blocks(&self.values),
                prefix: None,
            },
            payload,
            index: (),
            prefix: (),
            _marker: PhantomData,
        }
    }
}

impl<'l, D: MaterializableScalar, Chip: M, TableAxes: M, B: Backend> HbmTensorView<'l, D, Chip, TableAxes, B> {
    /// Starts a gather relative to this view's selected window.
    #[primitive(HbmTensor::gather)]
    pub fn gather<IndexedAxis: M, Payload: M>(self) -> GatherPlan<'l, D, Chip, TableAxes, IndexedAxis, Begin, B> {
        let payload = validate_gather_payload(&TableAxes::to_value(), &IndexedAxis::to_value(), &Payload::to_value())
            .unwrap_or_else(|error| panic!("{error}"));
        GatherPlan {
            table: GatherTable {
                source: TableSource::View(self),
                prefix: None,
            },
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
    /// Limits each batch's sparse positions by its own runtime length.
    #[primitive(GatherPlan::prefix)]
    pub fn prefix<Sparse: M, Lengths: M>(mut self, lengths: &'l HbmTensor<i32, Chip, Lengths, B>, offset: i32) -> Self {
        self.table.prefix = Some(BatchPrefix {
            lengths: lengths.memory(),
            batch: Pair::<Chip, Lengths>::to_value(),
            sparse: Sparse::to_value(),
            offset,
        });
        self
    }

    fn materialize<Cluster: M, Slice: M, Element: M>(
        self,
        dma: &DmaContext<{ Dma::Tensor }, B>,
        prefix: Option<usize>,
    ) -> DmTensor<D, Chip, Cluster, Slice, Element, B>
    where
        D: MaterializableScalar,
    {
        validate_placement::<Cluster, Slice, Element>(&self.payload, &self.index.domain);
        let (storage, mapping, indexed) = match self.table.source {
            TableSource::Tensor(tensor) => (
                tensor.read(dma).inner,
                Pair::<Chip, TableAxes>::to_value(),
                IndexedAxis::to_value(),
            ),
            TableSource::View(view) => (
                view.read(dma, |view| view.read().inner),
                Pair::<Chip, TableAxes>::to_value(),
                IndexedAxis::to_value(),
            ),
            TableSource::Blocks(blocks) => (
                slice::storage(dma, blocks),
                slice::mapping::<Chip, TableAxes>(blocks.len()),
                slice::outer::<Chip, TableAxes>(blocks.len()).pair(IndexedAxis::to_value()),
            ),
            TableSource::Window(blocks) => (
                slice::from_blocks::<D, Chip, TableAxes, B>(
                    blocks
                        .iter()
                        .map(|block| block.read(dma, |block| block.read().into_vec()))
                        .collect(),
                ),
                slice::mapping::<Chip, TableAxes>(blocks.len()),
                slice::outer::<Chip, TableAxes>(blocks.len()).pair(IndexedAxis::to_value()),
            ),
        };
        let index = self.index.tensor.read(dma);
        let lengths = self
            .table
            .prefix
            .as_ref()
            .map(|prefix| dma.load::<i32>(prefix.lengths.clone()));
        let mut output = DmTensor::from_parts(Tensor::zeroed());
        B::gather_table::<D, m![{ Chip }, { Cluster }, { Slice }, { Element }], Source::Axes>(
            &storage,
            &mut output.inner.inner,
            &index.inner,
            &mapping,
            &indexed,
            &Chip::to_value(),
            crate::backend::GatherRead {
                domain: &self.index.domain,
                prefix,
                unit: self.index.unit,
                batch_prefix: self
                    .table
                    .prefix
                    .as_ref()
                    .zip(lengths.as_ref())
                    .map(|(prefix, lengths)| crate::backend::GatherPrefix {
                        lengths,
                        batch: prefix.batch.clone(),
                        sparse: prefix.sparse.clone(),
                        offset: prefix.offset,
                    }),
            },
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
        assert!(
            self.table.prefix.is_none(),
            "combining sparse_prefix and per-batch prefix is not yet supported"
        );
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
    use crate::backend::Cpu;
    use crate::{Device, Topology};

    #[test]
    fn prefixes_count_tokens_across_entries() {
        axes![TE = 6, TB = 2, TW = 2, TR = 2, TC = 8];

        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 4 }).unwrap();
        let cell = |entry: i32, token: i32, lane: i32| entry * 100 + token * 10 + lane;
        let values = (0..6).flat_map(|e| (0..2).flat_map(move |b| (0..2).map(move |w| cell(e, b, w))));
        let table = HbmTensor::<i32, m![1], m![TE, TB, TW], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec(values.collect::<Vec<_>>()),
        );
        let entries = [[5, 1, 3, 0], [2, 4, 0, 1]];
        let stride = i32::try_from(TB::SIZE * TW::SIZE * std::mem::size_of::<i32>()).unwrap();
        let index = HbmTensor::<i32, m![1], m![TR, TC / 2], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec(
                entries
                    .as_flattened()
                    .iter()
                    .map(|entry| entry * stride)
                    .collect::<Vec<_>>(),
            ),
        );
        let lengths = HbmTensor::<i32, m![1], m![TR], Cpu>::from_value(&device.tdma, &Tensor::from_vec([5, 8]));

        let output: DmTensor<i32, m![1], m![1 # 2], m![[TR, TC / 2] # 64], m![TB, TW], Cpu> = table
            .gather::<m![TE], m![TB, TW]>()
            .by_byte_offsets(&index)
            .prefix::<m![TC], _>(&lengths, 2)
            .to_dm(&mut device.tdma);

        let expected = entries.iter().zip([2, 3]).flat_map(|(row, live)| {
            row.iter().enumerate().flat_map(move |(position, &entry)| {
                (0..2).flat_map(move |b| (0..2).map(move |w| if position < live { cell(entry, b, w) } else { 0 }))
            })
        });
        assert_eq!(
            output
                .to_hbm::<m![TR, TC / 2, TB, TW]>(&mut device.tdma)
                .load()
                .into_vec(),
            expected.collect::<Vec<_>>()
        );
    }
    axes![A = 5, K = 8, D = 4, V0 = 2, V1 = 3];

    #[test]
    fn output_mapping_accepts_permutation_and_factorization() {
        let payload =
            validate_gather_payload(&<m![K, D]>::to_value(), &<m![K]>::to_value(), &<m![D]>::to_value()).unwrap();

        validate_placement::<m![V1], m![V0], m![D]>(&payload, &<m![V0, V1]>::to_value());
    }

    /// Each chip gathers from its own table through its own index list.
    ///
    /// The values differ per chip on purpose: with the same table and the same offsets on every
    /// chip, a gather that read another chip's rows would produce the same answer and no comparison
    /// could see it.
    #[test]
    fn a_chip_gathers_through_its_own_table() {
        axes![PC = 2, PK = 2, PD = 2, PG = 2];

        let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 4 }).unwrap();
        // chip c, row k, lane d -> c*100 + k*10 + d
        let values = (0..PC::SIZE)
            .flat_map(|c| (0..PK::SIZE).flat_map(move |k| (0..PD::SIZE).map(move |d| (c * 100 + k * 10 + d) as i32)))
            .collect::<Vec<_>>();
        let table = HbmTensor::<i32, m![PC], m![PK, PD], Cpu>::from_value(&device.tdma, &Tensor::from_vec(values));
        let stride = (PD::SIZE * std::mem::size_of::<i32>()) as i32;
        // chip 0 reads rows [1, 0]; chip 1 reads rows [0, 1].
        let index =
            HbmTensor::<i32, m![PC], m![PG], Cpu>::from_value(&device.tdma, &Tensor::from_vec([stride, 0, 0, stride]));

        let output: DmTensor<i32, m![PC], m![1 # 2], m![[PG, PD] # 64], m![1], Cpu> = table
            .gather::<m![PK], m![PD]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        let output = output.to_hbm::<m![PG, PD]>(&mut device.tdma);

        assert_eq!(output.load().into_vec(), vec![10, 11, 0, 1, 100, 101, 110, 111]);
    }

    #[test]
    fn byte_offsets_use_the_payload_stride() {
        axes![TA = 2, TK = 3, TD = 2, TG = 2];

        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 4 }).unwrap();
        let values = (0..TA::SIZE)
            .flat_map(|a| (0..TK::SIZE).flat_map(move |k| (0..TD::SIZE).map(move |d| (a * 100 + k * 10 + d) as i32)))
            .collect::<Vec<_>>();
        let table = HbmTensor::<i32, m![1], m![TA, TK, TD], Cpu>::from_value(&device.tdma, &Tensor::from_vec(values));
        let stride_bytes = (TD::SIZE * std::mem::size_of::<i32>()) as i32;
        let index = HbmTensor::<i32, m![1], m![TG], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec([stride_bytes, 2 * stride_bytes]),
        );

        let output: DmTensor<i32, m![1], m![1 # 2], m![[TA, TG, TD] # 64], m![1], Cpu> = table
            .gather::<m![TK], m![TA, TD]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        let output = output.to_hbm::<m![TA, TG, TD]>(&mut device.tdma);

        assert_eq!(output.load().into_vec(), vec![10, 11, 20, 21, 110, 111, 120, 121]);
    }

    /// A gather whose output is replicated on both clusters writes the rows into each cluster.
    #[test]
    fn fills_each_replicated_cluster() {
        axes![TK = 4, TD = 4, TG = 2];

        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 4 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![TK, TD], Cpu>::from_value(&device.tdma, &Tensor::from_vec(0..16));
        let stride_bytes = (TD::SIZE * std::mem::size_of::<i32>()) as i32;
        let index = HbmTensor::<i32, m![1], m![TG], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec([3 * stride_bytes, stride_bytes]),
        );

        let output: DmTensor<i32, m![1], m![2], m![TD / 2 # 64], m![TG, TD % 2], Cpu> = table
            .gather::<m![TK], m![TD]>()
            .by_byte_offsets::<m![TG]>(&index)
            .to_dm(&mut device.tdma);

        let slice = |half: i32| [12 + 2 * half, 13 + 2 * half, 4 + 2 * half, 5 + 2 * half];
        let cluster = [slice(0), slice(1)]
            .concat()
            .into_iter()
            .chain(std::iter::repeat_n(0, 62 * 4));
        let expected = cluster.clone().chain(cluster).collect::<Vec<_>>();
        assert_eq!(output.inner.inner.to_vec(), expected);
    }

    #[test]
    fn prefixes_use_each_chips_length() {
        axes![PC = 2, PK = 2, PD = 2, PG = 2];
        let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 4 }).unwrap();
        let table = HbmTensor::<i32, m![PC], m![PK, PD], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec([0, 1, 10, 11, 100, 101, 110, 111]),
        );
        let indices = HbmTensor::<i32, m![PC], m![PG], Cpu>::from_value(&device.tdma, &Tensor::from_vec([8, -1, 0, 8]));
        let lengths = HbmTensor::<i32, m![PC], m![1], Cpu>::from_value(&device.tdma, &Tensor::from_vec([1, 2]));

        let output: DmTensor<i32, m![PC], m![1 # 2], m![[PG, PD] # 64], m![1], Cpu> = table
            .gather::<m![PK], m![PD]>()
            .by_byte_offsets(&indices)
            .prefix::<m![PG], _>(&lengths, 0)
            .to_dm(&mut device.tdma);

        assert_eq!(
            output.to_hbm::<m![PG, PD]>(&mut device.tdma).load().into_vec(),
            [10, 11, 0, 0, 100, 101, 110, 111]
        );
    }

    #[test]
    fn sparse_prefix_executes_the_requested_length_on_the_host() {
        axes![TK = 4, TD = 2, TG = 4];

        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 4 }).unwrap();
        let table = HbmTensor::<i32, m![1], m![TK, TD], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec([0, 1, 10, 11, 20, 21, 30, 31]),
        );
        let stride_bytes = (TD::SIZE * std::mem::size_of::<i32>()) as i32;
        let index = HbmTensor::<i32, m![1], m![TG], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec([0, stride_bytes, 2 * stride_bytes, 3 * stride_bytes]),
        );

        let output: DmTensor<i32, m![1], m![1 # 2], m![[TG, TD] # 64], m![1], Cpu> = table
            .gather::<m![TK], m![TD]>()
            .by_byte_offsets(&index)
            .sparse_prefix(1)
            .to_dm(&mut device.tdma);
        let output = output.to_hbm::<m![TG, TD]>(&mut device.tdma);

        assert_eq!(output.load().into_vec(), vec![0, 1, 0, 0, 0, 0, 0, 0]);
    }
}
