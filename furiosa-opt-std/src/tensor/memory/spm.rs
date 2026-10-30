//! Tensors held in a cluster's scratchpad memory.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::constraints;
use crate::runtime::CurrentBackend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

/// Tensor stored in scratchpad memory. `Pe` places the payload across the cluster's fused PEs.
#[primitive(SpmTensor)]
#[derive(Debug)]
pub struct SpmTensor<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend = CurrentBackend> {
    pub(super) inner: Tensor<D, Pair<Chip, Pair<Cluster, Pair<Pe, Element>>>, B>,
    _marker: PhantomData<(D, Chip, Cluster, Pe, Element)>,
}

impl<D: Scalar, Chip: M, Cluster: M, Pe: M, Element: M, B: Backend> SpmTensor<D, Chip, Cluster, Pe, Element, B> {
    /// Logical SPM mapping, including the explicit fused PE axis.
    pub type Mapping = m![{ Chip }, { Cluster }, { Pe }, { Element }];

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
