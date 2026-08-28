//! Fetch Engine: DM → Tensor Unit stream.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::engine::{CanApplyFetch, CanApplyFetchChipLift, CanApplyFetchClusterLift, CanApplyFetchSliceLift};
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

use furiosa_opt_lower::{FetchError, FetchLiftDimension, config_fetch, config_fetch_lift_mapping};

/// After the Fetch Engine (sequencer), before any adapter stage.
#[derive(Debug)]
pub struct PositionFetch;

impl Position for PositionFetch {}

/// Tensor streamed after the Fetch Engine.
pub type FetchTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetch, D, Chip, Cluster, Slice, Time, Packet, B>;

/// After a lift onto `Chip`; a lift onto `Cluster` or `Slice` may still follow.
#[derive(Debug)]
pub struct PositionFetchChipLift;

impl Position for PositionFetchChipLift {}

/// After a lift onto `Cluster`; a lift onto `Slice` may still follow.
#[derive(Debug)]
pub struct PositionFetchClusterLift;

impl Position for PositionFetchClusterLift {}

/// After a lift onto `Slice`, the innermost dimension a lift can address.
#[derive(Debug)]
pub struct PositionFetchSliceLift;

impl Position for PositionFetchSliceLift {}

/// Tensor streamed after a lift onto `Chip`.
pub type FetchChipLiftTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchChipLift, D, Chip, Cluster, Slice, Time, Packet, B>;

/// Tensor streamed after a lift onto `Cluster`.
pub type FetchClusterLiftTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchClusterLift, D, Chip, Cluster, Slice, Time, Packet, B>;

/// Tensor streamed after a lift onto `Slice`.
pub type FetchSliceLiftTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionFetchSliceLift, D, Chip, Cluster, Slice, Time, Packet, B>;

fn new_fetch_lift_tensor<
    'l,
    const T: Tu,
    P: Position,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
>(
    ctx: &'l mut TuContext<{ T }>,
    inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Packet>>>>, B>,
) -> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B> {
    TuTensor {
        ctx,
        inner,
        _position: PhantomData,
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchChipLiftTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        new_fetch_lift_tensor(ctx, inner)
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchClusterLiftTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        new_fetch_lift_tensor(ctx, inner)
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchSliceLiftTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        new_fetch_lift_tensor(ctx, inner)
    }
}

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    FetchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    #[doc(hidden)]
    pub fn new(ctx: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self {
            ctx,
            inner,
            _position: PhantomData,
        }
    }
}

// ANCHOR: fetch_impl
impl<'l, const T: Tu, P: CanApplyFetch, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Fetch Sequencer.
    #[primitive(TuTensor::fetch)]
    pub fn fetch<OutTime: M, OutPacket: M>(self) -> FetchTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_fetch::<Cluster, Slice, Time, Packet, OutTime, OutPacket>();
        FetchTensor::new(self.ctx, self.inner.transpose(true))
    }
}
// ANCHOR_END: fetch_impl

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    // ANCHOR: fetch_chip_lift
    /// Gives each chip its own fetch base, lifting a DM axis onto `Chip`.
    #[primitive(TuTensor::fetch_chip_lift)]
    pub fn fetch_chip_lift<OutChip: M, OutTime: M>(
        self,
    ) -> FetchChipLiftTensor<'l, T, D, OutChip, Cluster, Slice, OutTime, Packet, B>
    where
        P: CanApplyFetchChipLift,
    {
        // ANCHOR_END: fetch_chip_lift
        constraints::assert_chip_preserved::<Chip, OutChip>();
        constraints::assert_lift_factor::<Chip, Time, OutTime>();
        verify_fetch_lift::<Chip, OutChip, Time, Packet, OutTime>(FetchLiftDimension::Chip);
        FetchChipLiftTensor::new(self.ctx, self.inner.transpose(true))
    }
}

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    // ANCHOR: fetch_cluster_lift
    /// Gives each cluster its own fetch base, lifting a DM axis onto `Cluster`.
    #[primitive(TuTensor::fetch_cluster_lift)]
    pub fn fetch_cluster_lift<OutCluster: M, OutTime: M>(
        self,
    ) -> FetchClusterLiftTensor<'l, T, D, Chip, OutCluster, Slice, OutTime, Packet, B>
    where
        P: CanApplyFetchClusterLift,
    {
        // ANCHOR_END: fetch_cluster_lift
        constraints::assert_cluster_preserved::<Cluster, OutCluster>();
        constraints::assert_lift_factor::<Cluster, Time, OutTime>();
        verify_fetch_lift::<Cluster, OutCluster, Time, Packet, OutTime>(FetchLiftDimension::Cluster);
        FetchClusterLiftTensor::new(self.ctx, self.inner.transpose(true))
    }
}

impl<'l, const T: Tu, P: Position, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    // ANCHOR: fetch_slice_lift
    /// Gives each slice its own fetch base, lifting a DM axis onto `Slice`.
    #[primitive(TuTensor::fetch_slice_lift)]
    pub fn fetch_slice_lift<OutSlice: M, OutTime: M>(
        self,
    ) -> FetchSliceLiftTensor<'l, T, D, Chip, Cluster, OutSlice, OutTime, Packet, B>
    where
        P: CanApplyFetchSliceLift,
    {
        // ANCHOR_END: fetch_slice_lift
        constraints::assert_slice_preserved::<Slice, OutSlice>();
        constraints::assert_lift_factor::<Slice, Time, OutTime>();
        verify_fetch_lift::<Slice, OutSlice, Time, Packet, OutTime>(FetchLiftDimension::Slice);
        FetchSliceLiftTensor::new(self.ctx, self.inner.transpose(true))
    }
}

/// Validates the Fetch dimensions and synthesizes its read descriptors.
fn verify_fetch<Cluster: M, Slice: M, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    assert!(
        matches!(Cluster::SIZE, 1 | 2),
        "{}",
        FetchError::ClusterSize(Cluster::SIZE)
    );
    assert!(
        matches!(Slice::SIZE, 64 | 128 | 256),
        "{}",
        FetchError::SliceSize(Slice::SIZE)
    );
    // Switch validates the post-adapter packet width; Collect pads it to one flit.
    let _ = config_fetch(
        &Time::to_value(),
        &Packet::to_value(),
        &OutTime::to_value(),
        &OutPacket::to_value(),
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

/// Validates one placement lift and its `(time, packet, base)` descriptors.
fn verify_fetch_lift<Placement: M, OutPlacement: M, Time: M, Packet: M, OutTime: M>(dimension: FetchLiftDimension) {
    let lifted = config_fetch_lift_mapping(dimension, &Placement::to_value(), &OutPlacement::to_value())
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !lifted.idents().is_empty(),
        "this fetch lifts no axis; use a plain `fetch`"
    );
    // The packet a lift hands on is the one it was given: only `OutTime` loses the lifted axes.
    let packet = Packet::to_value();
    let _ = config_fetch(&Time::to_value(), &packet, &OutTime::to_value(), &packet, Some(&lifted))
        .unwrap_or_else(|e| panic!("{dimension} lift: {e}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![A = 8, B = 8, S = 64, X = 8, L = 16];

    // verify_fetch::<D, Cluster, Slice, Time, Packet, OutTime, OutPacket>; The DM form is the input
    // `Time ⊗ Packet`; `(OutTime, OutPacket)` is the requested stream.

    #[test]
    fn valid_read() {
        verify_fetch::<m![1], m![S], m![1], m![A, B], m![A], m![B]>();
    }
    /// Splits `L`: packet reads `L % 8`, while time walks `L / 8`.
    #[test]
    fn split_axis_across_packet_and_time() {
        verify_fetch::<m![1], m![S], m![1], m![L], m![L / 8], m![L % 8]>();
    }

    /// Replicates the packet across four stride-zero time steps.
    #[test]
    fn broadcast_time_read() {
        verify_fetch::<m![1], m![S], m![1], m![A], m![4], m![A]>();
    }

    #[test]
    fn live_input_time_reads() {
        verify_fetch::<m![1], m![S], m![A], m![B], m![A], m![B]>();
    }
    // verify_fetch_lift::<Placement, OutPlacement, Time, Packet, OutTime>; A lift stage moves axes
    // out of its own `Time` and into `OutPlacement`; the packet is handed on.

    #[test]
    fn valid_lift() {
        verify_fetch_lift::<m![S, 2], m![S, A / 4], m![A], m![B], m![A % 4]>(FetchLiftDimension::Slice);
    }

    #[test]
    fn valid_two_broadcast_lift() {
        verify_fetch_lift::<m![2, S, 2], m![A / 4, S, B / 4], m![A / 4, B / 4], m![X], m![1]>(
            FetchLiftDimension::Slice,
        );
    }
}
