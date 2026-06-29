//! Stream Adapter: stream-side validation and transpose for [`super::contract_outer`].
//!
//! - [`verify_stream_adapter`]: checks that the input `Time`/`Packet` can be
//!   expanded into the joint computation mapping (inner-flit packet match,
//!   Time-divides-OutTime, broadcast axes confined to innermost positions).
//! - [`contract_outer`]: transposes the stream tensor into the joint shape.

use furiosa_mapping::*;

use crate::backend::Backend;
use crate::engine::FLIT_BYTES;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

pub(super) fn contract_outer<
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Lane: M,
    Time: M,
    Packet: M,
    OutTime: M,
    OutPacket: M,
    B: Backend,
>(
    inner: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Time }, { Packet }], B>,
) -> Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { OutTime }, { OutPacket }], B> {
    verify_stream_adapter::<D, Lane, Time, Packet, OutTime, OutPacket>();
    inner.transpose(true)
}

/// Validates the stream-side mapping transformation for [`super::contract_outer`].
///
/// `OutPacket` collects `PackSize` flits (`∈ {1, 2}`), and packing absorbs the
/// innermost `PackSize` cells of `Time` into the packet. The joint mapping is then:
/// - `OutPacket = [packed, inner flit]`, where the inner flit equals the input
///   `Packet` and `packed` equals the innermost `PackSize` cells of `Time`,
///   padding included (a padded packed cell must have come from padding in `Time`).
/// - `OutTime = [outer, broadcast]`, where `outer` is the rest of `Time` and the
///   broadcast (tiling) axes occupy the innermost positions.
pub(super) fn verify_stream_adapter<D: Scalar, Lane: M, Time: M, Packet: M, OutTime: M, OutPacket: M>() {
    assert!(
        [1, 2, 4, 8].contains(&Lane::SIZE),
        "Lane::SIZE must be 1, 2, 4, or 8, got {}",
        Lane::SIZE
    );

    let out_packet_bytes = D::size_in_bytes_from_length(OutPacket::SIZE);
    assert!(
        out_packet_bytes.is_multiple_of(FLIT_BYTES) && [1, 2].contains(&(out_packet_bytes / FLIT_BYTES)),
        "OutPacket must be 1 or 2 flits ({FLIT_BYTES} bytes each), got {out_packet_bytes} bytes"
    );

    // Packing collects `pack_size` flits, pulling the innermost `pack_size` cells
    // of `Time` into the packet.
    let pack_size = out_packet_bytes / FLIT_BYTES;
    let (time_outer, time_packed) = Time::to_value().split_at(pack_size);

    // OutPacket = [packed cells of Time, inner flit]. The inner flit is the input
    // Packet; the packed cells must match Time's innermost cells exactly, padding
    // included.
    let flit_elements = D::length_from_bytes(FLIT_BYTES);
    let (out_packet_packed, out_packet_flit) = OutPacket::to_value().split_at(flit_elements);
    assert_eq!(
        out_packet_flit.normalize(),
        Packet::to_value().normalize(),
        "`contract_outer`: inner flit of OutPacket must equal the input Packet",
    );
    assert_eq!(
        out_packet_packed.normalize(),
        time_packed.normalize(),
        "`contract_outer`: OutPacket's packed cells must equal the innermost {pack_size} cells of Time",
    );

    // OutTime = [outer Time, broadcast]. The broadcast (tiling) axes are innermost,
    // so stripping them off the bottom must leave exactly the outer Time.
    assert!(
        OutTime::SIZE.is_multiple_of(time_outer.size()),
        "`contract_outer`: Time does not divide OutTime"
    );
    let tiling_size = OutTime::SIZE / time_outer.size();
    let (out_time_outer, _broadcast) = OutTime::to_value().split_at(tiling_size);
    assert_eq!(
        out_time_outer.normalize(),
        time_outer.normalize(),
        "`contract_outer`: OutTime must be the outer Time with broadcast (tiling) axes innermost",
    );
}
