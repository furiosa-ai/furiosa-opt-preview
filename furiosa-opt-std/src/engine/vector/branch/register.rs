use furiosa_mapping::{M, MappingExt};

use crate::engine::vector::scalar::VeScalar;
use crate::engine::vector::tensor::VeTensorShape;
use crate::tensor::Tensor;
use crate::tensor::memory::DmTensor;

#[derive(Debug)]
/// A scalar-register payload, independent of the pass's register-file port.
pub enum Register<D: VeScalar, Mapping: M> {
    /// A compile-time scalar shared by the stream.
    Constant(D),
    /// A runtime SRAM scalar broadcast within each slice.
    Scalar(Tensor<D, Mapping>),
}

impl<D: VeScalar, Mapping: M> Clone for Register<D, Mapping> {
    fn clone(&self) -> Self {
        match self {
            Self::Constant(value) => Self::Constant(*value),
            Self::Scalar(tensor) => Self::Scalar(tensor.clone()),
        }
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    Register<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
{
    pub(crate) fn scalar<Element: M>(tensor: &DmTensor<D, Chip, Cluster, Slice, Element>) -> Self {
        assert!(
            Element::to_value().axes().is_empty(),
            "a scalar SRAM operand must be constant within each slice"
        );
        Self::Scalar(tensor.inner.transpose(true))
    }
}
