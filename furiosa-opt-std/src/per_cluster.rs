//! Tensor arguments projected to each device chip and cluster, with scalar access on PE 0.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::runtime::Device;
use crate::scalar::{RuntimeIndex, RuntimeScalar, Scalar};
use crate::spm_scalar::SpmScalar;
use crate::tensor::memory::SpmTensor;
use crate::tensor::memory::spm::assert_in_payload;

/// A chip and cluster coordinate within the device running [`Device::per_cluster`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterCoordinate {
    chip: usize,
    cluster: usize,
    num_cluster: usize,
}

impl ClusterCoordinate {
    /// The chip-major cluster index within the device.
    fn linear_index(self) -> usize {
        self.chip * self.num_cluster + self.cluster
    }
}

/// Context passed to a `per_cluster` body.
/// It has no operations yet; per-cluster engine control will be reached through it.
#[primitive(ClusterContext)]
#[derive(Debug)]
pub struct ClusterContext<'s, B: Backend> {
    _device: PhantomData<&'s mut Device<B>>,
}

/// Borrowed scalar access to one cluster's SPM payload.
/// `S` provides shared or mutable access to the parent tensor's storage.
#[primitive(SpmTensorLocal)]
#[derive(Debug)]
pub struct SpmTensorLocal<D: Scalar, Pe: M, Element: M, B: Backend, S> {
    storage: S,
    coordinate: ClusterCoordinate,
    _marker: PhantomData<fn() -> (D, Pe, Element, B)>,
}

impl<D: Scalar + RuntimeScalar, Pe: M, Element: M, B: Backend, S> SpmTensorLocal<D, Pe, Element, B, S> {
    /// The element offset in the parent's storage for a local payload index.
    fn element_offset(&self, index: impl RuntimeIndex) -> usize {
        let offset = index.into_usize();
        assert_in_payload::<Element>(offset);
        self.coordinate.linear_index() * Pe::SIZE * Element::SIZE + offset
    }
}

/// A readable [`SpmTensorLocal`].
pub type SpmTensorLocalRef<'s, D, Pe, Element, B> =
    SpmTensorLocal<D, Pe, Element, B, &'s <B as crate::backend::internal::Backend>::Storage<D>>;

/// A writable [`SpmTensorLocal`].
pub type SpmTensorLocalMut<'s, D, Pe, Element, B> =
    SpmTensorLocal<D, Pe, Element, B, &'s mut <B as crate::backend::internal::Backend>::Storage<D>>;

impl<D: Scalar + RuntimeScalar, Pe: M, Element: M, B: Backend, S> SpmTensorLocal<D, Pe, Element, B, S>
where
    S: Deref<Target = <B as crate::backend::internal::Backend>::Storage<D>>,
{
    /// Reads `index` from this cluster's payload.
    /// Requires `index < len()`; see [`SpmTensor::get`] for bounds-checking behavior.
    #[primitive(SpmTensorLocal::get)]
    pub fn get(&self, index: impl RuntimeIndex) -> D {
        B::spm_local_load(&self.storage, self.element_offset(index))
    }

    /// See [`SpmTensor::len`](crate::tensor::memory::SpmTensor::len).
    #[allow(
        clippy::len_without_is_empty,
        reason = "`len() == 0` reads better than a method for it"
    )]
    #[inline]
    pub fn len(&self) -> u32 {
        Element::SIZE as u32
    }
}

impl<D: Scalar + RuntimeScalar, Pe: M, Element: M, B: Backend, S> SpmTensorLocal<D, Pe, Element, B, S>
where
    S: DerefMut<Target = <B as crate::backend::internal::Backend>::Storage<D>>,
{
    /// Writes `value` at `index` in this cluster's payload.
    /// Requires `index < len()`; see [`SpmTensor::get`] for bounds-checking behavior.
    #[primitive(SpmTensorLocal::set)]
    pub fn set(&mut self, index: impl RuntimeIndex, value: D) {
        let offset = self.element_offset(index);
        B::spm_local_store(&mut self.storage, offset, value);
    }
}

/// Lifetime-indexed family of projected arguments.
// A separate family avoids an implied `Self: 'c` bound that would require static argument borrows.
#[doc(hidden)]
pub trait LocalFamily {
    /// The projected argument, borrowed for `'c`.
    type Local<'c>;
}

/// The family of a readable [`SpmTensorLocal`].
#[doc(hidden)]
#[derive(Debug)]
pub struct SpmReadFamily<D, Pe, Element, B>(PhantomData<fn() -> (D, Pe, Element, B)>);

/// The family of a writable [`SpmTensorLocal`].
#[doc(hidden)]
#[derive(Debug)]
pub struct SpmWriteFamily<D, Pe, Element, B>(PhantomData<fn() -> (D, Pe, Element, B)>);

impl<D: Scalar, Pe: M, Element: M, B: Backend> LocalFamily for SpmReadFamily<D, Pe, Element, B> {
    type Local<'c> = SpmTensorLocalRef<'c, D, Pe, Element, B>;
}

impl<D: Scalar, Pe: M, Element: M, B: Backend> LocalFamily for SpmWriteFamily<D, Pe, Element, B> {
    type Local<'c> = SpmTensorLocalMut<'c, D, Pe, Element, B>;
}

/// An SPM argument projected to the current chip and cluster by [`Device::per_cluster`].
pub trait PerCluster {
    /// The argument type received by each cluster's closure.
    type Family: LocalFamily;

    #[doc(hidden)]
    fn project(&mut self, coordinate: ClusterCoordinate) -> <Self::Family as LocalFamily>::Local<'_>;

    /// Panics unless this argument holds one location per device chip and cluster.
    #[doc(hidden)]
    fn assert_spans_device(&self, num_chip: usize, num_cluster: usize);
}

impl<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend> PerCluster
    for &SpmTensor<D, Chip, Cluster, Pe, Element, B>
{
    type Family = SpmReadFamily<D, Pe, Element, B>;

    fn project(&mut self, coordinate: ClusterCoordinate) -> <Self::Family as LocalFamily>::Local<'_> {
        SpmTensorLocal {
            storage: self.storage(),
            coordinate,
            _marker: PhantomData,
        }
    }

    fn assert_spans_device(&self, num_chip: usize, num_cluster: usize) {
        assert_spans_device::<Chip, Cluster>(num_chip, num_cluster);
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend> PerCluster
    for &mut SpmTensor<D, Chip, Cluster, Pe, Element, B>
{
    type Family = SpmWriteFamily<D, Pe, Element, B>;

    fn project(&mut self, coordinate: ClusterCoordinate) -> <Self::Family as LocalFamily>::Local<'_> {
        SpmTensorLocal {
            storage: self.storage_mut(),
            coordinate,
            _marker: PhantomData,
        }
    }

    fn assert_spans_device(&self, num_chip: usize, num_cluster: usize) {
        assert_spans_device::<Chip, Cluster>(num_chip, num_cluster);
    }
}

/// The family of a readable scalar projection (`D`).
#[doc(hidden)]
#[derive(Debug)]
pub struct ScalarReadFamily<D>(PhantomData<fn() -> D>);

impl<D: Scalar> LocalFamily for ScalarReadFamily<D> {
    type Local<'c> = D;
}

impl<D: Scalar + RuntimeScalar, Chip: M, Cluster: M, B: Backend> PerCluster for &SpmScalar<D, Chip, Cluster, B> {
    type Family = ScalarReadFamily<D>;

    fn project(&mut self, coordinate: ClusterCoordinate) -> <Self::Family as LocalFamily>::Local<'_> {
        self.values[coordinate.linear_index()]
    }

    fn assert_spans_device(&self, num_chip: usize, num_cluster: usize) {
        assert_spans_device::<Chip, Cluster>(num_chip, num_cluster);
    }
}

macro_rules! tuple_per_cluster {
    ($($name:ident),+) => {
        impl<$($name: LocalFamily),+> LocalFamily for ($($name,)+) {
            type Local<'c> = ($($name::Local<'c>,)+);
        }

        impl<$($name: PerCluster),+> PerCluster for ($($name,)+) {
            type Family = ($($name::Family,)+);

            fn project(&mut self, coordinate: ClusterCoordinate) -> <Self::Family as LocalFamily>::Local<'_> {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                ($($name.project(coordinate),)+)
            }

            fn assert_spans_device(&self, num_chip: usize, num_cluster: usize) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.assert_spans_device(num_chip, num_cluster);)+
            }
        }
    };
}
tuple_per_cluster!(A);
tuple_per_cluster!(A, B);
tuple_per_cluster!(A, B, C);
tuple_per_cluster!(A, B, C, D);

/// What a [`Device::per_cluster`] body returns at one chip and cluster.
/// `P` carries the `Chip` and `Cluster` mappings of the collected value; the destination type infers it.
// Implemented on the local type so a body that returns `()` needs no annotation.
pub trait LocalResult<P>: Sized {
    /// The device-wide value holding one result per chip and cluster.
    type Collected;

    /// Collects one result per chip and cluster, in chip-major order.
    #[doc(hidden)]
    fn collect(locals: Vec<Self>, num_chip: usize, num_cluster: usize) -> Self::Collected;
}

impl LocalResult<()> for () {
    type Collected = ();

    fn collect(_: Vec<()>, _: usize, _: usize) {}
}

// A body that always panics returns `!`, which inference would otherwise leave without an impl.
impl LocalResult<()> for ! {
    type Collected = ();

    fn collect(_: Vec<!>, _: usize, _: usize) {}
}

/// The placement of a collected [`SpmScalar`].
#[doc(hidden)]
#[derive(Debug)]
pub struct ScalarPlacement<Chip, Cluster, B>(PhantomData<fn() -> (Chip, Cluster, B)>);

impl<D: Scalar + RuntimeScalar, Chip: M, Cluster: M, B: Backend> LocalResult<ScalarPlacement<Chip, Cluster, B>> for D {
    type Collected = SpmScalar<D, Chip, Cluster, B>;

    fn collect(locals: Vec<D>, num_chip: usize, num_cluster: usize) -> Self::Collected {
        assert_spans_device::<Chip, Cluster>(num_chip, num_cluster);
        SpmScalar::from_vec(locals)
    }
}

macro_rules! tuple_local_result {
    ($(($local:ident, $placement:ident)),+) => {
        impl<$($local: LocalResult<$placement>, $placement),+> LocalResult<($($placement,)+)> for ($($local,)+) {
            type Collected = ($($local::Collected,)+);

            // A macro cannot make names, so each member's value and vector reuse its type names.
            #[allow(non_snake_case)]
            fn collect(locals: Vec<Self>, num_chip: usize, num_cluster: usize) -> Self::Collected {
                $(let mut $placement = Vec::with_capacity(locals.len());)+
                for ($($local,)+) in locals {
                    $($placement.push($local);)+
                }
                ($($local::collect($placement, num_chip, num_cluster),)+)
            }
        }
    };
}
tuple_local_result!((RA, PA), (RB, PB));
tuple_local_result!((RA, PA), (RB, PB), (RC, PC));
tuple_local_result!((RA, PA), (RB, PB), (RC, PC), (RD, PD));

impl<B: Backend> Device<B> {
    /// Runs the same `body` in each chip and cluster with locally projected arguments.
    /// `body` is `Fn`: locations share no mutable state. Tensor results leave through `args`, and
    /// the scalars `body` returns are collected into the [`SpmScalar`]s the destination type names.
    /// Borrows the whole device, so collective DMA and tensor-unit calls stay outside `body`.
    #[primitive(Device::per_cluster)]
    // ANCHOR: per_cluster
    pub fn per_cluster<A: PerCluster, P, R: LocalResult<P>>(
        &mut self,
        mut args: A,
        body: impl for<'c> Fn(&mut ClusterContext<'_, B>, <A::Family as LocalFamily>::Local<'c>) -> R,
    ) -> R::Collected {
        // ANCHOR_END: per_cluster
        // The host invokes the body once per chip and cluster, using each cluster's own slice.
        let (num_chip, num_cluster) = (self.num_chip(), self.num_cluster());
        args.assert_spans_device(num_chip, num_cluster);
        let mut locals = Vec::with_capacity(num_chip * num_cluster);
        for chip in 0..num_chip {
            for cluster in 0..num_cluster {
                let coordinate = ClusterCoordinate {
                    chip,
                    cluster,
                    num_cluster,
                };
                let mut ctx = ClusterContext { _device: PhantomData };
                locals.push(body(&mut ctx, args.project(coordinate)));
            }
        }
        R::collect(locals, num_chip, num_cluster)
    }
}

/// Checks an argument's or result's `Chip::SIZE` and `Cluster::SIZE` against the device's counts.
fn assert_spans_device<Chip: M, Cluster: M>(num_chip: usize, num_cluster: usize) {
    assert!(
        Chip::SIZE == num_chip && Cluster::SIZE == num_cluster,
        "a `per_cluster` argument or result spans {} chips of {} clusters, and the device has {num_chip} of {num_cluster}",
        Chip::SIZE,
        Cluster::SIZE,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use crate::runtime::Topology;
    use crate::tensor::Tensor;

    axes![C = 2, E = 4];
    type Values = SpmTensor<i32, m![1], m![C], m![1], m![E], Cpu>;

    fn device() -> Device<Cpu> {
        Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap()
    }

    fn values() -> Values {
        Values::new(Tensor::from_vec([1, 2, 3, 4, 10, 20, 30, 40]))
    }

    #[test]
    fn a_returned_scalar_keeps_each_cluster_value() {
        let values = values();
        let sum: SpmScalar<i32, m![1], m![C], Cpu> =
            device().per_cluster(&values, |_, values| values.get(0u32) + values.get(1u32));
        assert_eq!(sum.values, [3, 30]);
    }

    #[test]
    fn a_returned_tuple_collects_each_member() {
        let values = values();
        let (first, last): (SpmScalar<i32, m![1], m![C], Cpu>, SpmScalar<i32, m![1], m![2], Cpu>) =
            device().per_cluster(&values, |_, values| (values.get(0u32), 7));
        assert_eq!(first.values, [1, 10]);
        assert_eq!(last.uniform_value(), 7);
    }
}
