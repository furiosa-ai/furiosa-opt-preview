use furiosa_mapping::M;
use furiosa_opt_common_ir::ImmSlot;

use super::{IntoBranchedOperand, IntoGroupOperand, IntoTernaryOperand, Plain, RegPayload, assert_fill, sealed};
use crate::engine::vector::branch::{BinaryBranchedOperand, Register, RfPort, TagGuard, TernaryBranchedOperand};
use crate::engine::vector::scalar::VeScalar;
use crate::engine::vector::tensor::VeTensorShape;
use crate::tensor::memory::DmTensor;

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> sealed::Sealed
    for &DmTensor<D, Chip, Cluster, Slice, Element>
{
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> Plain for &DmTensor<D, Chip, Cluster, Slice, Element> {}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M> sealed::SlotSealed
    for &DmTensor<D, Chip, Cluster, Slice, Element>
{
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, Time: M, Packet: M>
    RegPayload<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>> for &DmTensor<D, Chip, Cluster, Slice, Element>
where
    D: RegPayload<
            D,
            VeTensorShape<Chip, Cluster, Slice, Time, Packet>,
            Value = Register<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
            Port = RfPort<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        >,
{
    type Value = Register<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>;
    type Canonical = D;
    type Port = RfPort<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>;

    fn into_value(self) -> Self::Value {
        Register::scalar(self)
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, Time: M, Packet: M>
    IntoBranchedOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
    for &DmTensor<D, Chip, Cluster, Slice, Element>
{
    fn fill_slots(
        self,
        operand: &mut BinaryBranchedOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        guard: TagGuard,
    ) {
        assert_fill(operand.fill_imm(ImmSlot::Reg0, guard, Register::scalar(self)));
    }
}

impl<D: VeScalar, Chip: M, Cluster: M, Slice: M, Element: M, Time: M, Packet: M>
    IntoGroupOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
    for &DmTensor<D, Chip, Cluster, Slice, Element>
{
    fn into_group_operand(self) -> Option<BinaryBranchedOperand<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>> {
        Some(self.into_branched_operand())
    }
}

impl<Chip: M, Cluster: M, Slice: M, Element: M, Time: M, Packet: M>
    RegPayload<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
    for (&DmTensor<f32, Chip, Cluster, Slice, Element>, f32)
{
    type Value = (Register<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>, f32);
    type Canonical = (f32, f32);
    type Port = (RfPort<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>, f32);

    fn into_value(self) -> Self::Value {
        (Register::scalar(self.0), self.1)
    }
}

impl<Chip: M, Cluster: M, Slice: M, Element: M, Time: M, Packet: M>
    IntoTernaryOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>
    for (&DmTensor<f32, Chip, Cluster, Slice, Element>, f32)
{
    fn fill_ternary_slots(
        self,
        operand: &mut TernaryBranchedOperand<f32, VeTensorShape<Chip, Cluster, Slice, Time, Packet>>,
        guard: TagGuard,
    ) {
        assert_fill(operand.fill_imm(ImmSlot::Reg0, guard, (Register::scalar(self.0), self.1)));
    }
}
