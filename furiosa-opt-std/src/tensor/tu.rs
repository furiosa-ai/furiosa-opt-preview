//! Tensor Unit stream typestate primitive.
//!
//! Defines [`TuTensor<P>`], the in-motion tensor type whose [`Position`]
//! marker `P` tracks the Tensor Unit pipeline stage at compile time.
//!
//! Each engine ([`crate::engine`]) owns the typestate it *produces*: its
//! [`Position`] marker, its `XTensor` type alias, and the transition methods
//! that consume the source typestate. The exception is the pipeline entry
//! point ([`PositionBegin`] / [`BeginTensor`]), which is not produced by any
//! engine and lives here.
//!
//! `PositionVectorFinal` / `VectorFinalTensor` live in
//! [`crate::engine::vector::tensor`], produced by the Vector Engine.

use std::marker::PhantomData;

use crate::engine::vector::filtered_packets::FilteredPackets;
use furiosa_mapping::*;

use crate::backend::Backend;
use crate::constraints;
use crate::context::*;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;

/// Marker trait for pipeline position of Tensor Unit tensors.
pub trait Position: std::fmt::Debug + 'static {}

/// After beginning the pipeline.
#[derive(Debug)]
pub struct PositionBegin;

impl Position for PositionBegin {}

/// How far along `Time` each slice of a stream carries packets.
pub trait TimeExtent: std::fmt::Debug + sealed::Sealed {}

/// Every slice carries all of its `Time` packets.
#[derive(Debug)]
pub struct FixedPackets;

impl TimeExtent for FixedPackets {}
impl sealed::Sealed for FixedPackets {}

/// A tag filter's result: `Time` is a bound, and each slice carries a runtime prefix of it, which
/// `FilteredPackets` records.
#[derive(Debug)]
pub struct TagFiltered(pub(crate) FilteredPackets);

impl TimeExtent for TagFiltered {}
impl sealed::Sealed for TagFiltered {}

/// Tensor streamed through the Tensor Unit pipeline.
#[derive(Debug)]
pub struct TuTensor<
    'l,
    const T: Tu,
    P: Position,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend = CurrentBackend,
    E: TimeExtent = FixedPackets,
> {
    pub(crate) device: &'l mut TuContext<{ T }>,
    pub(crate) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Slice, Pair<Time, Packet>>>>, B>,
    pub(crate) time_extent: E,
    pub(crate) _position: PhantomData<P>,
}

impl<
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
    E: TimeExtent,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B, E>
{
    /// Mapping type alias.
    pub type Mapping = m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }];
}

/// Tensor streamed after the beginning.
pub type BeginTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionBegin, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    BeginTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
    }

    #[doc(hidden)]
    pub fn new(device: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            device,
            inner,
            time_extent: FixedPackets,
            _position: PhantomData,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}
