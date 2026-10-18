use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use super::vector_tensor::{apply_unary_op_where, create_inter_slice_reduce_tensor};
use super::{VeTensorShape, VectorBranchTensor, VectorInitTensor, VectorInterSliceReduceTensor};
use crate::context::Tu;
use crate::engine::vector::branch::Register;
use crate::engine::vector::operand::{IntoBranchedOperand, NoStash, OneReg, VeOperandBuilder};
use crate::engine::vector::stage::markers::VeOrder;
use crate::engine::vector::stash_slot::Fresh;

/// One constant VRU operand, optionally constrained by an execution-ID predicate.
pub trait ScaleOperand<Shape: M>: IntoBranchedOperand<f32, Shape> + sealed::Sealed {}

impl<Shape: M> ScaleOperand<Shape> for f32 {}

impl<Shape: M> ScaleOperand<Shape> for VeOperandBuilder<f32, Shape, f32, NoStash, OneReg> {}

mod sealed {
    use super::*;

    pub trait Sealed {}

    impl Sealed for f32 {}

    impl<Shape: M> Sealed for VeOperandBuilder<f32, Shape, f32, NoStash, OneReg> {}
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet>
{
    /// Scales each element in the VRU without reducing its coordinates.
    #[primitive(VectorInitTensor::vector_scale)]
    pub fn vector_scale(
        self,
        factor: f32,
    ) -> VectorInterSliceReduceTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, { VeOrder::InterFirst }> {
        create_inter_slice_reduce_tensor(self.device, self.inner.map(|value| value * factor))
    }
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorBranchTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Fresh, { VeOrder::IntraFirst }>
{
    /// Scales matching execution IDs in the VRU and preserves every other element.
    #[primitive(VectorTensor::vector_scale)]
    pub fn vector_scale<Op>(
        self,
        factor: Op,
    ) -> VectorInterSliceReduceTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, { VeOrder::IntraFirst }>
    where
        Op: ScaleOperand<VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
    {
        let operand = factor.into_branched_operand();
        let [Some((guard, Register::Constant(factor))), None, None] = operand.reg_slots() else {
            unreachable!("a scale operand fills exactly one immediate register");
        };
        let (device, inner, tag, state) = self.into_parts();
        let scaled = apply_unary_op_where(&inner, &tag, *guard, |value| value * factor);
        VectorInterSliceReduceTensor::from_parts(device, scaled, tag, state)
    }
}
