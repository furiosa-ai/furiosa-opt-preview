//! SPM scalars distributed over chips and clusters.

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use std::marker::PhantomData;

use crate::backend::Backend;
use crate::runtime::CurrentBackend;
use crate::scalar::{RuntimeScalar, Scalar};

/// One value of `D` on PE 0 at each active chip and cluster.
#[primitive(SpmScalar)]
#[derive(Debug)]
pub struct SpmScalar<D: Scalar, Chip: M, Cluster: M, B: Backend = CurrentBackend> {
    /// One value per (Chip, Cluster), in chip-major order.
    pub(crate) values: Vec<D>,
    _marker: PhantomData<fn() -> (Chip, Cluster, B)>,
}

impl<D: Scalar + RuntimeScalar, Chip: M, Cluster: M, B: Backend> SpmScalar<D, Chip, Cluster, B> {
    /// Creates a scalar holding `value` at every active chip and cluster.
    #[primitive(SpmScalar::uniform)]
    pub fn uniform(value: D) -> Self {
        Self::from_vec(vec![value; Chip::SIZE * Cluster::SIZE])
    }

    /// One value per (Chip, Cluster), in cell order.
    pub(crate) fn from_vec(values: impl IntoIterator<Item = D>) -> Self {
        let values: Vec<D> = values.into_iter().collect();
        assert_eq!(
            values.len(),
            Chip::SIZE * Cluster::SIZE,
            "a device scalar holds one value per (Chip, Cluster)"
        );
        Self {
            values,
            _marker: PhantomData,
        }
    }
}

/// Scalar values broadcast across both chips and clusters.
impl<D: Scalar + RuntimeScalar, const CHIPS: usize, const CLUSTERS: usize, B: Backend>
    SpmScalar<D, Broadcast<CHIPS>, Broadcast<CLUSTERS>, B>
{
    /// Reads the shared value, checking on the host that all stored copies agree.
    #[primitive(SpmScalar::uniform_value)]
    pub fn uniform_value(&self) -> D {
        let first = *self.values.first().expect("a device scalar has at least one location");
        assert!(
            self.values.iter().all(|value| *value == first),
            "a uniform scalar holds different values at different locations"
        );
        first
    }
}
