//! The Vector Engine's Filter stage, which `vector_filter_by_tag` and `vector_filter_by_bitmap` end in.
//!
//! A bitmap filter fills every packet, so its stream is `FixedPackets`. A tag filter's slices emit a
//! runtime prefix of `Time`, so its stream is `TagFiltered`, a [`TimeExtent`] every later stage keeps.

use std::marker::PhantomData;

use furiosa_mapping::*;
use furiosa_opt_common_ir::CollectBitmap;
use furiosa_opt_common_ir::{ExecutionId, TagGuard};

use crate::engine::vector::filtered_packets::{FilteredPackets, PacketCount};
use furiosa_opt_macro::primitive;

use crate::backend::Backend;
use crate::context::*;
use crate::engine::vector::cast::{VectorCast, VectorCastTensor, cast_stream};
use crate::engine::vector::layer::Reinterpret;
use crate::engine::vector::op::semantics::HasConversionOp;
use crate::engine::vector::scalar::VeScalar;
use crate::engine::vector::stage::markers as stage;
use crate::engine::vector::tensor::VectorFinalTensor;
use crate::runtime::CurrentBackend;
use crate::scalar::*;
use crate::tensor::Tensor;
use crate::tensor::tu::{FixedPackets, Position, TagFiltered, TimeExtent, TuTensor};

/// Stream position right after a filter: the Vector Engine's `Filter` stage, as
/// [`PositionVectorCast`](crate::engine::vector::cast::PositionVectorCast) is its `Cast` stage.
pub type PositionVectorFilter = stage::Filter;

impl Position for PositionVectorFilter {}

/// Tensor streamed after a filter.
pub type VectorFilterTensor<
    'l,
    const T: Tu,
    D,
    Chip,
    Cluster,
    Slice,
    Time,
    Packet,
    B = CurrentBackend,
    E = FixedPackets,
> = TuTensor<'l, { T }, PositionVectorFilter, D, Chip, Cluster, Slice, Time, Packet, B, E>;

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend, E: TimeExtent>
    VectorFilterTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B, E>
{
    /// Casts and packs each packet.
    #[primitive(VectorTensor::vector_cast)]
    pub fn vector_cast<OutD: Scalar, OutPacket: M>(
        self,
    ) -> VectorCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, OutPacket, B, E>
    where
        D: VectorCast<OutD>,
    {
        cast_stream(self.device, self.inner, self.time_extent)
    }

    /// Exits the Vector Engine.
    #[primitive(VectorTensor::vector_final)]
    pub fn vector_final(self) -> VectorFinalTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B, E> {
        VectorFinalTensor::new_with_time_extent(self.device, self.inner, self.time_extent)
    }

    /// Rereads the stream's bits as `D2`. Emits no instruction, so it keeps the position.
    #[primitive(VectorTensor::vector_reinterpret)]
    pub fn vector_reinterpret<D2: VeScalar>(
        self,
    ) -> VectorFilterTensor<'l, T, D2, Chip, Cluster, Slice, Time, Packet, B, E> {
        let op_fn = HasConversionOp::<D, D2>::conversion_op_fn(&Reinterpret);
        TuTensor {
            inner: self.inner.map(&op_fn),
            device: self.device,
            time_extent: self.time_extent,
            _position: PhantomData,
        }
    }
}

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    VectorFilterTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Keeps the packet positions `bitmap` names out of each slice's `input_elements`. A filter reads
    /// and emits packets of one size, so `Packet` is also the input's packet.
    pub(crate) fn keep(
        device: &'l mut TuContext<{ T }>,
        data: &[D],
        input_elements: usize,
        bitmap: CollectBitmap,
    ) -> Self {
        // The device fills the rest of the last packet with zeros (Arena 329087).
        let output = keep_positions(
            data,
            input_elements,
            Time::SIZE * Packet::SIZE,
            Packet::SIZE,
            bitmap,
            D::from_raw_bits(0),
        );
        Self {
            device,
            inner: Tensor::from_vec(output),
            time_extent: FixedPackets,
            _position: PhantomData,
        }
    }
}

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorFilterTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, CurrentBackend, TagFiltered>
{
    /// Packs the elements of each slice's `input_elements` whose tag `guard` admits and records its
    /// emitted packet count.
    pub(crate) fn select(
        device: &'l mut TuContext<{ T }>,
        data: &[D],
        tags: &[ExecutionId],
        input_elements: usize,
        guard: TagGuard,
    ) -> Self {
        // The device fills the rest of the last packet with zeros (Arena 326119).
        let (output, packets) = pack_admitted(
            data,
            tags,
            input_elements,
            Time::SIZE * Packet::SIZE,
            Packet::SIZE,
            guard,
            D::from_raw_bits(0),
        );
        Self {
            device,
            inner: Tensor::from_vec(output),
            time_extent: TagFiltered(FilteredPackets::new(packets, PacketCount(Time::SIZE))),
            _position: PhantomData,
        }
    }
}

/// Packs the elements whose tag `guard` admits out of each slice's `input_elements` into its
/// `output_elements`, the rest of which is `padding`, and counts the packets of `packet_elements`
/// each slice emits, past `output_elements` too.
fn pack_admitted<V: Copy>(
    data: &[V],
    tags: &[ExecutionId],
    input_elements: usize,
    output_elements: usize,
    packet_elements: usize,
    guard: TagGuard,
    padding: V,
) -> (Vec<V>, Vec<PacketCount>) {
    assert_eq!(data.len(), tags.len(), "every element carries one tag");
    let slices = data.len() / input_elements;
    let mut output = vec![padding; slices * output_elements];
    let mut packets = Vec::with_capacity(slices);
    let streams = data
        .chunks_exact(input_elements)
        .zip(tags.chunks_exact(input_elements))
        .zip(output.chunks_exact_mut(output_elements));
    for ((data, tags), output) in streams {
        let mut survivors = 0;
        for (&value, &tag) in data.iter().zip(tags) {
            if guard.admits(tag) {
                if let Some(slot) = output.get_mut(survivors) {
                    *slot = value;
                }
                survivors += 1;
            }
        }
        packets.push(PacketCount(survivors.div_ceil(packet_elements)));
    }
    (output, packets)
}

/// Keeps the positions `bitmap` names of every `packet_elements` packet out of each slice's
/// `input_elements` and lays them out in its `output_elements`, the rest of which is `padding`.
fn keep_positions<V: Copy>(
    data: &[V],
    input_elements: usize,
    output_elements: usize,
    packet_elements: usize,
    bitmap: CollectBitmap,
    padding: V,
) -> Vec<V> {
    let mut output = Vec::with_capacity(data.len() / input_elements * output_elements);
    for input in data.chunks_exact(input_elements) {
        let survivors = input
            .chunks_exact(packet_elements)
            .flat_map(|packet| bitmap.positions().map(|position| packet[position]));
        let end = output.len() + output_elements;
        output.extend(survivors);
        assert!(output.len() <= end, "the output must hold every survivor");
        output.resize(end, padding);
    }
    output
}
