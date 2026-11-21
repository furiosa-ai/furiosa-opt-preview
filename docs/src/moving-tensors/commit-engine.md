# Commit Engine

The Commit Engine writes a Tensor Unit stream packet to DM, the inverse of the [Fetch Engine](./fetch-engine.md), after adapters have selected the final element type and layout.
It is the Commit Sequencer: a [mathematical tensor move](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move) that runs independently in every slice, each writing to its own local DM partition.

Choose Commit when a computed Tensor Unit stream must become a DM tensor with a deliberate `Element` layout.

## Interface

A `TuTensor` carries `Chip`, `Cluster`, `Slice`, `Time`, and `Packet` dimensions at the end of the Tensor Unit pipeline.
Its `Time` reflects the temporal unrolling of the computation, and `Packet` is the element layout in the output stream.

`.commit()` writes the stream to a new `DmTensor` in DM, and `.commit_view(destination)` writes it to an existing mutable DM view.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/commit.rs:commit_impl}}
```

`.commit()` preserves the `Chip`, `Cluster`, and `Slice` dimensions unchanged, because each slice independently writes to its own DM partition.
The output `Element` mapping replaces `Time` and `Packet`, defining how the stream is laid out in DM.
`Element` configures both the Commit Sequencer and the [Commit Adapter](../computing-tensors/commit-adapter.md), and can reorder `Time` axes relative to the input stream, performing a transpose during the commit.
For performance implications of the `Element` mapping, see [Optimizations](#optimizations).

The following example commits a cast accumulation result to DM as `bf16`.
The output `DmTensor` stores 16 time steps × 8 `bf16` elements across 256 slices.
Here `D = bf16` and `Element = m![M, N # 16]`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, M = 16, N = 8];

fn cast_commit<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1 # 2], m![P], m![M], m![N]>,
) -> DmTensor<bf16, m![1], m![1 # 2], m![P], m![M, N # 16]> {
    // Cast f32 to bf16 (Vector Engine cast compaction), then commit to DM (Commit Engine).
    // Input: M = 16 time steps, N = 8 f32 elements per packet (32 bytes).
    // After cast: N = 8 bf16 elements padded to 16 (32 bytes).
    // After trim: N # 16 trimmed into N = 8.
    // The sequencer writes across P = 256 slices.
    input.vector_init().vector_cast::<bf16, m![N # 16]>().vector_final().commit_trim::<m![N]>().commit()
}
#
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
#
# let c: ContractTensor<'_, _, f32, m![1], m![1 # 2], m![P], m![M], m![N]> = ContractTensor::new(&mut device.main, Tensor::zero());
# let _o = cast_commit(c);
#
# Ok(())
# }
```

## Slice placement

A commit writes each stream slice to the same slice of the destination.
`commit_view(destination)` therefore writes all slices or only part of them, and never moves data between slices.

Use `slice_tile()` on a mutable destination view to write only the slices inside the tile and preserve the others.
The following function writes the slices with `B < 4` and preserves those with `B >= 4`:

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 8, P = 8];

fn copy_first_half<'l>(
    device: &'l mut Device,
    source: DmTensorView<'l, i32, m![1], m![1], m![A, B], m![P]>,
    destination: DmTensorViewMut<'l, i32, m![1], m![1], m![A, B], m![P]>,
) {
    let destination = destination.slice_tile::<m![B / 4], 1, m![A, B % 4 #{!} 8]>(0);
    device.main.begin(source).fetch::<m![1], m![P]>().collect::<m![1], m![P]>()
        .commit_trim::<m![P]>().commit_view(destination);
}
```

The stream from `source` holds a value in all 64 slices, while the mutable tile marks the destination slices with `B >= 4` as `#{!}` padding.
The commit drops the stream values that land on `#{!}` padding, so the destination may hold fewer valid slices than the stream.
The compiler derives a slice bitmap from that padding, one bit per slice, and the commit skips the slices whose bit is off.
Destination `#` padding is written with don't-care values; only `#{!}` padding is skipped.

For a stream `Slice` placement `m![A, B, C]` with `C = 8`, the following destination forms are supported:

| Destination `Slice` | Slices the commit writes |
| --- | --- |
| `m![A, B, C]` | All |
| `m![A, B, C = 4 #{!} 8]` | `C < 4` |
| `m![A, B, 1 #{!} 8]` | `C = 0` |
| `m![A, B, 1 # 8]` | All, and the slices with `C > 0` hold don't-care values |

At every non-padding destination slice, the stream and destination `Slice` placements must produce the same `Index`, the logical coordinates of that slice.
Coordinates are compared per axis; an axis absent from a placement has coordinate 0.

Reordering `Slice` axes is rejected, as in this call:

```rust,ignore
axes![A = 8, B = 8, P = 8];
fn moved_slices<'l>(
    stream: CommitTrimTensor<'l, { Tu::Main }, i32,
        m![1], m![1], m![A, B], m![1], m![P]>,
    destination: DmTensorViewMut<'l, i32, m![1], m![1], m![B, A], m![P]>,
) {
    stream.commit_view(destination); // Rejected: the Slice placements disagree at slice 1.
}
```

With `A = 8` and `B = 8`, slice 1 has coordinates `A = 0, B = 1` in the stream but `A = 1, B = 0` in the destination.

Destination `#{0}` padding promises zero, so only stream `#{0}` padding may reach it.
A stream slice that holds a value is rejected there; pad the destination with `#` or `#{!}` instead.

Stream `#` and `#{!}` padding has no defined value, so the corresponding destination slice must also use `#` or `#{!}` padding.
A stream from a tiled source therefore cannot be written to an untiled destination:

```rust,ignore
axes![A = 8, B = 8, P = 8];
fn untiled_destination<'l>(
    stream: CommitTrimTensor<'l, { Tu::Main }, i32,
        m![1], m![1], m![A, B % 4 # 8], m![1], m![P]>,
    destination: DmTensorViewMut<'l, i32, m![1], m![1], m![A, B], m![P]>,
) {
    stream.commit_view(destination); // Rejected: the slices with B >= 4 hold no value.
}
```

`slice_tile()` currently supports only the constant start index `0`.
A start index moves the window along the tiled axis: `m![A = 2 #{!} 8]` at start index 1 writes the slices with `1 <= A < 3`.
A nonzero or runtime start index is not implemented yet.
Until then the compiler reports it as unsupported, and the CPU backend panics on a nonzero one.

```rust,ignore
axes![A = 8, B = 8, P = 8];
fn windows<'l>(source: DmTensorView<'l, i32, m![1], m![1], m![A, B], m![P]>, start: usize) {
    source.slice_tile::<m![B / 4], 1, m![A, B % 4 # 8]>(1);     // Unsupported: nonzero start index.
    source.slice_tile::<m![B / 4], 1, m![A, B % 4 # 8]>(start); // Unsupported: runtime start index.
}
```

## Constraints

- **Hardware dimensions**: `Chip::SIZE`, `Cluster::SIZE`, and `Slice::SIZE` must match the hardware configuration (see [Spatial and Temporal Dimensions](../mapping-tensors/spatial-temporal-dimensions.md#constraints)).
- **Address alignment**: All sequencer strides must be multiples of 8 bytes.
- **Write unit alignment**: `D[valid_size]` must be 8, 16, 24, or 32 bytes (see the [Commit Adapter's Trimming](../computing-tensors/commit-adapter.md#trimming) stage).

## Multi-Write Packet

Writing a packet may require multiple hardware writes because packet axes may not be contiguous in DM.
The compiler derives `write_size = gcd(valid_size, access_size)`.
The [Commit Adapter](../computing-tensors/commit-adapter.md) supplies `valid_size`; the [Sequencer](./sequencer.md#access-size) supplies `access_size`.
In the [sub-context](../computing-tensors/index.md#execution-context), `D[write_size]` is fixed at 8 bytes.
The total cycle count is `Time::SIZE * (valid_size / write_size)`.
The division is always exact: in the main-context, `valid_size == write_size`, so each packet commits in a single cycle.
In the sub-context, `write_size` is fixed at 8 bytes and `valid_size` is one of 8, 16, 24, or 32 bytes (from the trimming constraint), so `valid_size / write_size` is always 1, 2, 3, or 4.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16, L = 32];

// Compiler-generated configuration: [
//   M -> 4 : 64,  (64 == 2 * 32,  contiguous)
//   K -> 2 : 32,   (32  == 32 * 1,  contiguous)
//   M -> 32 : 1    (packet dimension, contiguous)
// ] : 8
// access_size = 64; valid_size = 8; write_size = gcd(64, 8) = 8; writes per packet = 1
fn no_transpose<'l, const T: Tu>(
    input: VectorFinalTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![L]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![M, K, L]> {
    input.commit_trim::<m![L]>().commit()
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   W -> 8 : 1    (packet dimension, contiguous)
// ] : 32
// access_size = 8; valid_size = 8; write_size = gcd(8, 8) = 8; writes per packet = 1
fn transpose<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![M, K], m![W]>,
) -> DmTensor<f32, m![1], m![1 # 2], m![1 # 256], m![K, M, W]> {
    input.commit_trim::<m![W]>().commit()
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   N -> 8 : 1    (trimmed packet dimension, contiguous)
// ] : 16
// access_size = 8; valid_size = 8 (trimmed from 16); write_size = gcd(8, 8) = 8; writes per packet = 1
fn transpose_with_trimming<'l, const T: Tu>(
    input: VectorFinalTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![N # 32]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![K, M, N]> {
    input.commit_trim::<m![N]>().commit()
}

#
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# let a: VectorFinalTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![L]> = VectorFinalTensor::new(&mut device.main, Tensor::zero());
# let _o = no_transpose(a);
# let b: ContractTensor<'_, _, f32, m![1], m![1 # 2], m![1 # 256], m![M, K], m![W]> = ContractTensor::new(&mut device.main, Tensor::zero());
# let _o = transpose(b);
# let c: VectorFinalTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![N # 32]> = VectorFinalTensor::new(&mut device.main, Tensor::zero());
# let _o = transpose_with_trimming(c);
#
# Ok(())
# }
```

## Optimizations

Three factors determine Commit Sequencer throughput.

- **Sequential Addresses**: Writing to sequential DM addresses within each slice enables parallel bank access (128 B/cycle per DMN, 256 B/cycle with DMN interleaving).
  Patterns that hit the same bank 64+ times consecutively trigger [DM Bank Starvation](./memory-performance.md#bank-starvation).
- **Spatial parallelism**: Distributing writes across all active slices maximizes throughput.
- **Aligned writes** (invariant): Partial bank writes never occur, because both the write address and the write unit are always 8-byte aligned.
  Sequencer strides are multiples of 8 bytes (see [Constraints](#constraints)), and the [Commit Adapter's Trimming](../computing-tensors/commit-adapter.md#trimming) stage holds `D[valid_size]` to a multiple of 8 bytes.
