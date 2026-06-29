//! Hardware constraints shared across engine tensor constructors.
//!
//! Each engine's `Tensor::new` validates its inputs against these. The values
//! and checks live here, in one place, so the rules are stated once and every
//! engine refers to the same definitions instead of repeating literals.

/// Supported `Cluster` dimension sizes.
pub(crate) const CLUSTER_SIZES: [usize; 2] = [1, 2];

/// Bits in a byte.
pub(crate) const BITS_PER_BYTE: usize = 8;

/// Byte granularity a packet must align to in the [`Fetch`](crate::engine) and
/// [`Switch`](crate::engine) engines, which pass packets through unchanged.
pub(crate) const PACKET_ALIGN_BYTES: usize = 8;

/// Size of a single flit in bytes.
///
/// Data flows through the switching network in flit-sized units. The collect
/// engine and the compute engines normalize packets to exactly one flit.
pub(crate) const FLIT_BYTES: usize = 32;

/// Supported `Slice` dimension sizes, in bytes.
pub(crate) const SLICE_SIZES: [usize; 3] = [64, 128, 256];

/// Byte size of `length` elements of a `bits`-wide scalar.
///
/// `const fn` twin of [`Scalar::size_in_bytes_from_length`](crate::scalar::Scalar::size_in_bytes_from_length),
/// callable from the `const fn` constraint checks. Panics if the total bit count is not byte-aligned.
pub(crate) const fn size_in_bytes(bits: usize, length: usize) -> usize {
    assert!(
        (length * bits).is_multiple_of(BITS_PER_BYTE),
        "total bits must be byte-aligned"
    );
    (length * bits) / BITS_PER_BYTE
}

/// Asserts the `Cluster` dimension is one of the [`CLUSTER_SIZES`].
pub(crate) const fn assert_cluster_size(size: usize) {
    let mut i = 0;
    let mut ok = false;
    while i < CLUSTER_SIZES.len() {
        ok |= CLUSTER_SIZES[i] == size;
        i += 1;
    }
    assert!(ok, "Cluster size must be 1 or 2");
}

/// Asserts the `Slice` dimension is one of the [`SLICE_SIZES`].
pub(crate) const fn assert_slice_size(size: usize) {
    let mut i = 0;
    let mut ok = false;
    while i < SLICE_SIZES.len() {
        ok |= SLICE_SIZES[i] == size;
        i += 1;
    }
    assert!(ok, "Slice size must be one of 64 | 128 | 256");
}

/// Asserts a packet is a whole number of [`PACKET_ALIGN_BYTES`]-byte access words (Switch input).
pub(crate) const fn assert_packet_aligned_by_access_width(bits: usize, packet: usize) {
    assert!(
        size_in_bytes(bits, packet).is_multiple_of(PACKET_ALIGN_BYTES),
        "Packet size must be 8-byte aligned"
    );
}

/// Asserts a packet is a non-zero multiple of [`PACKET_ALIGN_BYTES`] bytes, at most one flit
/// (Commit trim output: 8 | 16 | 24 | 32 bytes).
pub(crate) const fn assert_packet_aligned_by_access_width_max_flit(bits: usize, packet: usize) {
    let bytes = size_in_bytes(bits, packet);
    assert!(
        bytes != 0 && bytes.is_multiple_of(PACKET_ALIGN_BYTES) && bytes <= FLIT_BYTES,
        "Packet size must be 8, 16, 24, or 32 bytes"
    );
}

/// Asserts a packet is exactly one flit ([`FLIT_BYTES`] bytes) (Collect, the compute engines, and
/// the Contraction lane output).
pub(crate) const fn assert_packet_one_flit(bits: usize, packet: usize) {
    assert!(
        size_in_bytes(bits, packet) == FLIT_BYTES,
        "Packet size must be exactly one flit (32 bytes)"
    );
}

/// Asserts a packet is one or two flits (Contraction `outer` output: 32 or 64 bytes).
pub(crate) const fn assert_packet_one_or_two_flit(bits: usize, packet: usize) {
    let bytes = size_in_bytes(bits, packet);
    assert!(
        bytes == FLIT_BYTES || bytes == 2 * FLIT_BYTES,
        "Packet size must be one or two flits (32 or 64 bytes)"
    );
}
