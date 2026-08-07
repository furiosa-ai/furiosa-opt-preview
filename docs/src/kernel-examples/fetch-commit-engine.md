# Fetch and Commit Engine

Fetch and Commit move a logical tensor between DM layouts while preserving every logical index.
The streaming path is input DM, Fetch, Switch, Collect, Commit, and output DM.
Collect and every stage after it consume a flit, which is a fixed 32-byte transfer unit, so packet size relative to flit size drives most of the decisions here.
This chapter covers four of those decisions: layout permutation, full-flit commits, packet padding, and capacity-bounded segmentation.
The [Fetch Engine](../moving-tensors/fetch-engine.md), [Collect Engine](../computing-tensors/collect-engine.md), and [Commit Engine](../moving-tensors/commit-engine.md) are the canonical API references.

## Data Movement Design Objectives

| Need | Design decision | Required evidence |
|------|-----------------|-------------------|
| Layout permutation | Preserve each logical index while changing its physical order. | A host oracle compares each logical index before and after movement. |
| Full-flit movement | Match packet and commit granularity to avoid unnecessary slicing. | Schedule and alignment constraints justify the chosen granularity. |
| Tail handling | Pad only when the padding cost is lower than the movement cost it avoids. | The oracle ignores padded values and validates the logical output shape. |
| Large tensors | Partition work into independently valid segments that fit the selected memory level. | The oracle covers segment boundaries and the recombined output. |

Every transformation begins by defining the logical input-to-output index relation.
`typecheck run` validates mapping and shape decisions, an emulation oracle validates values and boundaries, and the Schedule Viewer compares fixed candidates.

## Layout Permutation

Fetch and Commit permute the axes that the stream visits over time; they do not exchange the axis that sits innermost in a packet.
The runnable example below reads an `i8` tensor whose DM layout is `[A, B, C]` and writes the same values as `[B, A, C]`.
`C` remains innermost in both layouts because it is the packet axis, while `A` and `B` are `Time` axes whose visiting order the kernel chooses.
Exchanging the innermost axis instead, as in `[A, B]` to `[B, A]`, belongs to the [Transpose Engine](../computing-tensors/transpose-engine.md) within a flit or to a [tensor DMA](../moving-tensors/dma-engine.md) between memories.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 4, B = 8, C = 32];

fn fetch_and_commit_permuted<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![B, A, C]> {
    input
        .fetch::<m![B, A], m![C]>()
        .collect::<m![B, A], m![C # 32]>()
        .commit_trim::<m![C]>()
        .commit()
}
#
# let mut ctx = Context::acquire();
# let input: BeginTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _output = fetch_and_commit_permuted(input);
```

The input DM mapping keeps `C` innermost, so each Fetch packet is 32 contiguous `i8` values, which is both a legal packet size and exactly one flit.
`fetch::<m![B, A], m![C]>()` states the permutation directly: it streams `B × A = 32` time steps of one packet each, visiting `B` before `A` even though the input buffer stores `A` before `B`.
The Fetch Sequencer computes the strides for that order, so the read walks the input in steps of `B × C` bytes.
`collect` passes the 32-byte packet through unchanged, `commit_trim` keeps all 32 values, and `commit` writes them in stream order, which makes the output contiguous under `m![B, A, C]`.

A permutation changes which values are adjacent in memory without a Vector Engine or Contraction Engine operation, which prepares an operand for a later stage that needs a different visiting order.
An activation stored as `[batch, token, channel]`, for example, can be presented as `[token, batch, channel]` for a stage that iterates tokens outermost, because only the two outer axes move and `channel` stays innermost.
The current API derives both sequencer configurations from these mappings, so a kernel author writes no physical base address or commit size.

The same permutation can be expressed on the Commit side instead, by fetching `m![A, B]` and committing to `m![B, A, C]`.
Both produce the same output tensor and differ only in which sequencer absorbs the strided access: permuting at Fetch reads strided and writes contiguously, while permuting at Commit reads contiguously and writes strided.
Prefer the side whose access pattern the [Sequencer](../moving-tensors/sequencer.md) can express in fewer entries, and confirm the choice with a schedule comparison rather than by inspection.

### Permutation Dataflow

| Stage | Time | Packet | Effect |
|-------|------|--------|--------|
| Input DM | `1` | `A, B, C` | The Element mapping, which keeps `C` as a 32-byte contiguous tail. |
| Fetch | `B, A` | `C` | Reads one `C` packet per time step in `B`-major order, which strides the input. |
| Collect | `B, A` | `C # 32` | Passes the full flit through without padding or splitting. |
| Commit Trim | `B, A` | `C` | Keeps all 32 values, so trimming is a no-op. |
| Commit | `B, A` | `C` | Writes each packet contiguously, which realizes `m![B, A, C]`. |

The table below traces the first few time steps, using DM byte offsets within one slice.
Input offsets are `(a × B + b) × C` and output offsets are `(b × A + a) × C`, which is the stream position itself times `C`.

| Stream position | Index `[a, b]` | Input byte offset | Output byte offset |
|-----------------|----------------|-------------------|--------------------|
| `0` | `[0, 0]` | `0` | `0` |
| `1` | `[1, 0]` | `256` | `32` |
| `2` | `[2, 0]` | `512` | `64` |
| `3` | `[3, 0]` | `768` | `96` |
| `4` | `[0, 1]` | `32` | `128` |
| `31` | `[3, 7]` | `992` | `992` |

### Correctness Requirements

The host oracle must compare `output[b, a, c]` with `input[a, b, c]` for every logical index.
A test that checks only one time step, or only the total set of values, accepts a kernel that permutes the wrong axis pair.
`typecheck run` checks the mapping and shape path, while an emulation test checks the values that actually land in DM.
In RNGD, the generated sequencer must satisfy the current hardware limits of at most eight entries, at most 65,536 iterations per entry, and a packet size of 1, 2, 4, 8, 16, or 32 bytes.
Commit rejects a broadcast stream because every output tensor location must have exactly one source position.

The `fetch_commit_simple` example in `furiosa-opt-examples` changes an innermost axis instead, so it uses the other mechanism.
Its Tensor Unit stage fetches, casts to `i32`, and commits without changing the layout, and the `[A, B]` to `[B, A]` exchange comes from the output mapping of `to_hbm`, which the tensor DMA performs.
Its emulation test compares every `[B, A]` output index against the transposed and widened input.

## Full-Flit Commit

A full-flit write is a 32-byte output packet, not a separate Commit API.
The Collect Engine passes an exactly 32-byte packet through without padding or splitting it into more time steps.
`commit_trim` remains mandatory, but trimming a 32-byte valid packet is a no-op before `commit` writes it to DM.

Choose a full-flit packet when the logical packet and element type make 32 bytes natural for the next stage.
Do not add padding solely to make a packet 32 bytes without comparing the memory cost and schedule.
The [Collect Engine single-flit example](../computing-tensors/collect-engine.md#single-flit-packet) and [Commit Engine multi-write model](../moving-tensors/commit-engine.md#multi-write-packet) define the current behavior.

### Commit Granularity and Cost

An `i8` packet with `L = 32` occupies one complete flit with no Collect padding.
The code keeps `L` unchanged through `commit_trim`, so every byte in the flit is logical output rather than tail padding.
In the main context, the Commit Engine writes one valid packet per cycle when the mapping gives a contiguous 32-byte access.
In the sub context, Commit uses fixed eight-byte write units, so the same 32-byte packet takes four writes.
The output mapping still determines address order, so a full flit does not by itself make a non-contiguous layout efficient.

| Logical packet | Flits after Collect | Padding per logical packet | Commit consideration |
|----------------|---------------------|----------------------------|----------------------|
| 8 `i8` values | 1 × 32-byte flit | 24 bytes | A small packet consumes a full flit. |
| 16 `i8` values | 1 × 32-byte flit | 16 bytes | Padding may be acceptable when it reduces Fetch iterations. |
| 32 `i8` values | 1 × 32-byte flit | 0 bytes | The packet is a natural full-flit Commit candidate. |
| 40 `i8` values | 2 × 32-byte flits | 24 bytes | The final flit is padded. |

### Full-Flit Commit API

The following current API function commits an already-cast `i8` packet with `L = 32`.
The packet occupies one complete 32-byte flit, so `commit_trim` keeps all 32 values before `commit` writes the DM tensor.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, M = 4, K = 2, L = 32];

fn commit_full_flit<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1 # 2], m![P], m![M, K], m![L]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![P], m![M, K, L]> {
    input.commit_trim::<m![L]>().commit()
}
#
# let mut ctx = Context::acquire();
# let input: CastTensor<'_, _, i8, m![1], m![1 # 2], m![P], m![M, K], m![L]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _output = commit_full_flit(input);
```

| Boundary | Logical packet | Physical size | Result |
|----------|----------------|---------------|--------|
| Commit input | `L = 32` `i8` values | 32 bytes | One complete flit. |
| `commit_trim::<m![L]>()` | All 32 values | 32 bytes | No tail values are removed. |
| `commit()` | `M, K, L` | 32 bytes per packet | Writes the full logical packet to DM. |

## Packet Padding and Size

Padding is part of the tensor mapping, not an independent runtime `dummy` parameter.
The `#` mapping operator makes physical capacity available while leaving the logical tensor shape unchanged.
Fetch derives its `read_size` from contiguous DM access, packet size, and downstream cast constraints.
Larger packets can reduce fetch cycles, but Collect pads every non-32-byte packet to a 32-byte flit and can add time steps for multi-flit packets.

Use the [Fetch Engine optimization model](../moving-tensors/fetch-engine.md#optimizations) to compare candidate packet mappings.
Use the [Collect Engine](../computing-tensors/collect-engine.md) to determine the resulting flit and time mappings.
Use [`commit_trim`](../computing-tensors/commit-adapter.md#trimming) and a host oracle to ensure padding never appears in the logical output.
Compare candidates in the Schedule Viewer with the release, shapes, data types, and mappings fixed.

### Padding Performance Model

Fetch throughput depends on contiguous DM access and packet size.
Non-adjacent axes reduce `max_access_size` and therefore `read_size`, while a packet smaller than the contiguous run also limits `read_size`.
Padding can increase `read_size`, but the downstream Collect Engine converts packets to 32-byte flits.
A 20-byte packet wastes 12 of 32 flit bytes, and a 40-byte packet spans two flits and wastes 24 of 64 bytes.
Padding reads beyond logical data are safe only when those values do not affect computation.
The output path must still trim the padded tail before Commit materializes the logical tensor.

### Packet-Size Alternatives

The three current API functions below read the same `A = 3`, `B = 5`, `C = 2` `f8e4m3` tensor with different packet mappings.
Each value has one byte, so the packet mapping directly gives the packet byte size.

| Function | Fetch `Time` | Fetch `Packet` | Fetch work | Collect padding |
|----------|--------------|----------------|------------|-----------------|
| `fetch_packet_c` | `A, B` | `C # 8` | 15 packets of 8 bytes | 24 bytes for each packet. |
| `fetch_packet_bc` | `A` | `[B, C] # 16` | 3 packets of 16 bytes | 16 bytes for each packet. |
| `fetch_packet_abc` | `1` | `[A, B, C] # 32` | 1 packet of 32 bytes | None. |

The 8-byte and 16-byte candidates require Collect to pad each packet to a 32-byte flit.
The 32-byte candidate reaches Collect as one full flit, but its extra physical capacity can increase DM usage.
The schedule comparison decides whether fewer fetches outweigh the padding and the downstream layout cost for the fixed workload.

### Packet-Size API Examples

The current Fetch Engine examples express three packet sizes for the same `f8e4m3` tensor with `A = 3`, `B = 5`, and `C = 2`.
They process fifteen 8-byte packets, three 16-byte packets, or one 32-byte packet respectively before the downstream stages decide the final flit layout.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, B = 5, C = 2];

fn fetch_packet_c<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![A, B], m![C # 8]> {
    input.fetch()
}

fn fetch_packet_bc<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![A], m![[B, C] # 16]> {
    input.fetch()
}

fn fetch_packet_abc<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![[A, B, C] # 32]> {
    input.fetch()
}
#
# let mut ctx = Context::acquire();
# let input: BeginTensor<'_, _, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _packet_c = fetch_packet_c(input);
```

The larger packet is not automatically better because padding consumes DM capacity and can change the downstream dataflow.
Use the host oracle to prove that padding is excluded and the schedule to compare the fixed workload.

## Capacity-Bounded Segmentation

An operand that does not fit the register file it must live in is processed one segment at a time, and the kernel writes that loop itself.
The current API exposes no contract that automatically splits one Fetch and Commit operation into capacity-sized executions.

The example below adds a vector of `B = 4096` `i32` values to every row of a matrix with `A = 128` rows.
The Vector Engine reads its second operand from the vector register file (VRF), which holds 8 KiB per slice, and the whole vector occupies 16 KiB per slice, so the vector cannot be resident for a single pass.
The kernel therefore splits `B` into two segments of 2048 values, which is 8 KiB each.
Each iteration selects one segment with `tile`, loads only that segment with `to_vrf`, adds it to the matching input segment, and writes the result with `commit_view` before the next iteration overwrites the VRF.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 128, B = 4096];

type Chip = m![1];
type Cluster = m![1];

fn segmented_add(
    ctx: &mut Context,
    lhs: HbmTensorView<'_, i32, Chip, m![A, B]>,
    rhs: HbmTensorView<'_, i32, Chip, m![B]>,
    mut out: DmTensor<i32, Chip, Cluster, m![A / 2 # 64], m![A % 2, B]>,
) -> DmTensor<i32, Chip, Cluster, m![A / 2 # 64], m![A % 2, B]> {
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2 # 64], m![A % 2, B]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2 # 64], m![B]>(&mut ctx.tdma);

    for i in 0..2 {
        let rhs_tile = rhs_dm
            .view()
            .tile::<m![B / 2048], 1, m![B / 2048 = 1 # 2, B % 2048]>(i);
        let rhs_vrf: VrfTensor<i32, Chip, Cluster, m![A / 2 # 64], m![B % 2048]> = ctx
            .sub
            .begin(rhs_tile)
            .fetch::<m![B / 2048 = 1], m![B % 2048]>()
            .fetch_cast::<i32>()
            .collect::<m![B / 2048 = 1, B / 8 % 256], m![B % 8]>()
            .to_vrf();

        let lhs_tile = lhs_dm
            .view()
            .tile::<m![B / 2048], 1, m![A % 2, B / 2048 = 1 # 2, B % 2048]>(i);
        let out_tile = out
            .view_mut()
            .tile::<m![B / 2048], 1, m![A % 2, B / 2048 = 1 #{!} 2, B % 2048]>(i);
        ctx.main
            .begin(lhs_tile)
            .fetch::<m![A % 2, B / 2048 = 1], m![B % 2048]>()
            .fetch_cast::<i32>()
            .collect::<m![A % 2, B / 2048 = 1, B / 8 % 256], m![B % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &rhs_vrf)
            .vector_final()
            .commit_trim::<m![B % 8]>()
            .commit_view(out_tile);
    }

    out
}
```

### Segment Capacity Contract

Capacity is a per-slice constraint, so the segment size follows from what one slice must hold rather than from the total cluster capacity.
The `A / 2 # 64` mapping spreads 64 outer `A` positions across slices, and each slice keeps both `A % 2` positions of the segment it is working on.

| Per active slice | Values | Bytes | Where it must fit |
|------------------|--------|-------|-------------------|
| Whole `rhs_dm` vector | `4096` `i32` | 16 KiB | DM, which is why it cannot go to VRF in one pass. |
| One `rhs_vrf` segment | `2048` `i32` | 8 KiB | VRF, exactly at its per-slice capacity. |
| One `lhs_tile` | `2 × 2048` `i32` | 16 KiB | DM, read by Fetch. |
| One `out_tile` | `2 × 2048` `i32` | 16 KiB | DM, written by `commit_view`. |

| Loop iteration | Selected `B / 2048` position | Logical `B` indices | VRF content |
|----------------|------------------------------|---------------------|-------------|
| `0` | `0` | `0` through `2047` | The first vector segment. |
| `1` | `1` | `2048` through `4095` | The second vector segment, overwriting the first. |

`to_vrf` rejects an operand larger than the file, so this split is a contract the API enforces rather than a convention to remember.
Because one segment fills the VRF exactly, a kernel that also keeps a second vector operand resident must halve the segment again.
The segment count is therefore part of the kernel contract: changing it changes both the capacity that each pass demands and the schedule that the pass produces.
The Contraction Engine register file (8 KiB per lane) and DM (512 KiB per slice) bound their own operands the same way, so the same loop shape applies with a different capacity number.

### Segmentation Verification

Check that the segments cover every logical `B` index exactly once and that output segments do not overlap.
Check the host oracle across the segment boundary, because a correct first segment does not prove the second segment's offset or write mapping.
Compare schedule makespan after fixing the segment shape, data type, mappings, and release.
The [Tiling](./tiling.md) chapter explains the capacity decision, and the [Commit Engine](../moving-tensors/commit-engine.md) defines the write constraints.
