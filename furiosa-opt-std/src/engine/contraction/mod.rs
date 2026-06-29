//! Contraction Engine: Outer -> Packet Reducer -> Time Reducer -> Lane Folder.
//!
//! Submodules:
//! - [`outer`]: prepares operands in the joint computation mapping
//!   (`.contract_outer`).
//! - [`packet`]: Packet Reducer (`.contract_packet`).
//! - [`time`]: Time Reducer (`.contract_time`).
//! - [`lane`]: Lane Folder (`.contract_lane`).
//!
//! Output: a [`ContractTensor`], at [`PositionContraction`].

pub mod lane;
pub mod outer;
pub mod packet;
pub mod time;

use std::marker::PhantomData;

pub use lane::LaneMode;
pub use outer::ContractOuterTensor;

use furiosa_mapping::*;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// Number of columns in the temporal accumulator buffer.
pub(crate) const TEMPORAL_ACCUMULATOR_COLS: usize = 32;

/// Number of elements in the `contract_lane` output packet (`i32`/`f32`).
pub(crate) const CONTRACT_LANE_OUT_PACKET_ELEMENTS: usize = 8;

/// Asserts a reduce packet's element count is a power of two and at most
/// [`TEMPORAL_ACCUMULATOR_COLS`] (`packet_reduce` / `time_reduce` output).
const fn assert_packet_pow2_within_accumulator_cols(size: usize) {
    assert!(
        size != 0 && size & (size - 1) == 0 && size <= TEMPORAL_ACCUMULATOR_COLS,
        "Packet element count must be a power of two and at most the accumulator column count (32)"
    );
}

/// After the contraction engine.
#[derive(Debug)]
pub struct PositionContraction;

impl Position for PositionContraction {}

/// Intermediate tensor after the Packet Reducer (reduce-add within `Packet`),
/// before the Time Reducer.
#[derive(Debug)]
pub struct ContractPacketTensor<
    'l,
    const T: Tu,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
}

/// Intermediate tensor after the Time Reducer (accumulator buffer reduce across `Time`),
/// before the Lane Folder.
#[derive(Debug)]
pub struct ContractTimeTensor<
    'l,
    const T: Tu,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
> {
    pub(crate) ctx: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
    /// Pre-reduce `Time` mapping captured by `contract_time`, used by `contract_lane` to
    /// derive the cross-stage `inner_size` accumulator-buffer bound.
    pub(crate) pre_reduce_time: Mapping,
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractPacketTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    const fn check_constraints() {
        constraints::assert_cluster_size(Cluster::SIZE);
        constraints::assert_slice_size(Slice::SIZE);
        assert_packet_pow2_within_accumulator_cols(Packet::SIZE);
    }

    #[doc(hidden)]
    pub(crate) fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
    ) -> Self {
        const { Self::check_constraints() };
        Self { ctx, inner }
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    const fn check_constraints() {
        constraints::assert_cluster_size(Cluster::SIZE);
        constraints::assert_slice_size(Slice::SIZE);
        assert_packet_pow2_within_accumulator_cols(Packet::SIZE);
    }

    #[doc(hidden)]
    pub(crate) fn new(
        ctx: &'l mut TuContext<{ T }>,
        inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Lane, Pair<Time, Packet>>>>>, B>,
        pre_reduce_time: Mapping,
    ) -> Self {
        const { Self::check_constraints() };
        Self {
            ctx,
            inner,
            pre_reduce_time,
        }
    }
}

/// The padded extent of each axis of `m` keyed by its cumulative (dividend) stride, inner-to-outer.
///
/// The FMapping-free peer of the old factor walk (`stride -> padding_size/stride | resize`): on the
/// normalized `Mapping`, each axis already carries its padded extent as `size()` (a `Padding` node's
/// size is its padded total), and the running stride is the product of the inner axes' extents. The
/// contraction verifies key into this by a division term's `dividend_stride`.
pub(crate) fn padding_per_stride(m: &Mapping) -> std::collections::HashMap<usize, usize> {
    let mut extents = Vec::new();
    collect_axis_extents(&m.normalize(), &mut extents);
    let mut map = std::collections::HashMap::new();
    let mut stride = 1;
    for extent in extents {
        map.insert(stride, extent);
        stride *= extent;
    }
    map
}

/// Pushes each axis's padded extent (`size()`), innermost (right) first, down the normalized `Pair`
/// spine — the order the factor list used.
fn collect_axis_extents(m: &Mapping, out: &mut Vec<usize>) {
    match m {
        Mapping::Pair { left, right } => {
            collect_axis_extents(right, out);
            collect_axis_extents(left, out);
        }
        other => out.push(other.size()),
    }
}

/// Tensor streamed after the contraction engine.
pub type ContractTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionContraction, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    ContractTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    const fn check_constraints() {
        constraints::assert_cluster_size(Cluster::SIZE);
        constraints::assert_slice_size(Slice::SIZE);
        constraints::assert_packet_one_flit(D::BITS, Packet::SIZE);
    }

    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        const { Self::check_constraints() };

        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}
