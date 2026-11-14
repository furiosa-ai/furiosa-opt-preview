//! Vector Engine cast compaction.

use furiosa_mapping::*;
use furiosa_opt_lower::{CastKind, VectorCastInput, config_vector_cast};
use furiosa_opt_macro::primitive;
use std::marker::PhantomData;

use crate::backend::Backend;
use crate::cast::Cast;
use crate::constraints;
use crate::context::*;
use crate::engine::vector::scalar::VeScalar;
use crate::engine::vector::stage::markers as stage;
use crate::engine::vector::tensor::{VeTensorShape, VectorFinalTensor};
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{Position, TuTensor};

/// Supported narrowing conversions in the Vector Engine's cast-compaction stage.
#[diagnostic::on_unimplemented(
    message = "cannot apply `vector_cast()` from `{Self}` to `{D}`",
    label = "unsupported Vector Engine cast conversion",
    note = "supported conversions: i32 -> i4/i8/i16 and f32 -> f8e4m3/f8e5m2/bf16; use `.fetch_cast::<...>()` for widening"
)]
pub trait VectorCast<D: Scalar>: Cast<D> {
    /// Hardware conversion represented by this scalar pair.
    const KIND: CastKind;
}

impl VectorCast<i4> for i32 {
    const KIND: CastKind = CastKind::I32ToI4;
}
impl VectorCast<i8> for i32 {
    const KIND: CastKind = CastKind::I32ToI8;
}
impl VectorCast<i16> for i32 {
    const KIND: CastKind = CastKind::I32ToI16;
}
impl VectorCast<f8e4m3> for f32 {
    const KIND: CastKind = CastKind::F32ToF8E4M3;
}
impl VectorCast<f8e5m2> for f32 {
    const KIND: CastKind = CastKind::F32ToF8E5M2;
}
impl VectorCast<bf16> for f32 {
    const KIND: CastKind = CastKind::F32ToBf16;
}

/// Stream position after VE cast compaction.
pub type PositionVectorCast = stage::Cast;

impl Position for PositionVectorCast {}

/// Packed stream at the Vector Engine's cast-compaction stage.
pub type VectorCastTensor<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet, B = CurrentBackend> =
    TuTensor<'l, { T }, PositionVectorCast, D, Chip, Cluster, Slice, Time, Packet, B>;

impl<'l, const T: Tu, D: Scalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    VectorCastTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Finalizes the Vector Engine output.
    #[primitive(VectorTensor::vector_final)]
    pub fn vector_final(self) -> VectorFinalTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B> {
        VectorFinalTensor::new(self.device, self.inner)
    }

    fn check_constraints() {
        constraints::assert_cluster_size::<Cluster>();
        constraints::assert_slice_size::<Slice>();
        constraints::assert_packet_one_flit::<D, Packet>();
    }

    #[doc(hidden)]
    pub fn new(device: &'l mut TuContext<{ T }>, inner: Tensor<D, Self::Mapping, B>) -> Self {
        Self::check_constraints();

        Self {
            device,
            inner,
            _position: PhantomData,
        }
    }
}

pub(crate) fn cast_stream<
    'l,
    const T: Tu,
    D: VeScalar + VectorCast<OutD>,
    OutD: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    OutPacket: M,
    B: Backend,
>(
    device: &'l mut TuContext<{ T }>,
    input: Tensor<D, VeTensorShape<Chip, Cluster, Slice, Time, Packet>, B>,
) -> VectorCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, OutPacket, B> {
    verify_cast::<D, OutD, Packet, OutPacket>();
    VectorCastTensor::new(device, input.map(|value| value.cast()).transpose(false))
}

/// Validates the VE cast-compaction packet layouts.
fn verify_cast<D: VectorCast<OutD>, OutD: Scalar, InPacket: M, OutPacket: M>() {
    config_vector_cast(VectorCastInput {
        in_packet: InPacket::to_value(),
        out_packet: OutPacket::to_value(),
        kind: D::KIND,
    })
    .unwrap_or_else(|message| panic!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::CollectTensor;
    use crate::engine::vector::{TagMode, op::FxpBinaryOp};

    type Unit = m![1];
    type Slice = m![S];
    type Shape<Time, Packet> = crate::engine::vector::tensor::VeTensorShape<Unit, Unit, Slice, Time, Packet>;
    axes![P = 8, T = 2, S = 64];

    #[test]
    fn cast_only_stores_padded_narrow_values_in_vrf() {
        let mut sub = TuContext::<{ Tu::Sub }>::on();
        let input = Tensor::<i32, Shape<m![T], m![P]>>::from_vec((-8..8).cycle().take(64 * 16));
        let stream = CollectTensor::<{ Tu::Sub }, i32, Unit, Unit, Slice, m![T], m![P]>::new(&mut sub, input);
        let vrf = stream
            .vector_init()
            .vector_cast::<i4, m![P # 64]>()
            .vector_final()
            .to_vrf::<m![T, P # 64]>();
        let mut expected = vec![0; 64 * 2 * 64];
        for slice in 0..64 {
            for packet in 0..2 {
                for lane in 0..8 {
                    expected[(slice * 2 + packet) * 64 + lane] = (packet * 8 + lane) as i32 - 8;
                }
            }
        }
        assert_eq!(
            vrf.inner.into_vec().into_iter().map(i32::from).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn cast_after_arithmetic_stores_into_a_main_context_vrf_view() {
        let mut main = TuContext::<{ Tu::Main }>::on();
        let mut sub = TuContext::<{ Tu::Sub }>::on();
        let input =
            Tensor::<i32, Shape<Unit, m![P]>>::from_vec([-8, -6, -4, -2, 0, 2, 4, 6].into_iter().cycle().take(64 * 8));
        let stream = CollectTensor::<{ Tu::Main }, i32, Unit, Unit, Slice, Unit, m![P]>::new(&mut main, input);
        let mut vrf = crate::tensor::memory::VrfTensor::<i4, Unit, Unit, Slice, m![P # 64]>::new();
        stream
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, 1)
            .vector_cast::<i4, m![P # 64]>()
            .vector_final()
            .to_vrf_view(&mut sub, vrf.view_mut());
        let mut expected_packet = vec![0; 64];
        expected_packet[..8].copy_from_slice(&[-7, -5, -3, -1, 1, 3, 5, 7]);
        assert_eq!(
            vrf.inner.into_vec().into_iter().map(i32::from).collect::<Vec<_>>(),
            expected_packet.repeat(64)
        );
    }
}
