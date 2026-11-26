# Filter

The Vector Engine's Filter stage packs selected elements into eight-element packets within each slice, preserving their input order.
Choose a tag filter to select by a condition computed in the Intra-Slice Chain, or a bitmap filter to select the same positions in every input packet.

## Interface

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_filter_by_tag}}

{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_filter_by_bitmap}}
```

- `Time2` describes how output elements are arranged across packets in a slice.
- `Packet2` describes how output elements are arranged within each packet.
- `Time2::SIZE` is the packet capacity per slice.
  For example, an output with two eight-element packets has `Time2::SIZE = 2` and `Packet2::SIZE = 8`.
- The stream's *time extent* tells you how many packets each slice carries, starting at packet 0.

| Method | What it selects | Where to call it | Output time extent |
| --- | --- | --- | --- |
| [`vector_filter_by_tag(guard)`](#filter-by-tag) | Elements that pass the tag test `guard` | Intra-Slice Chain after tagging | `TagFiltered`. The packet count is determined at runtime for each slice. |
| [`vector_filter_by_bitmap::<BITMAP, _, _>()`](#filter-by-bitmap) | Positions whose bits are `1` in `BITMAP` | Intra-Slice Chain or Inter-Slice Reducer output | `FixedPackets` (the default). Each slice emits exactly `Time2::SIZE` packets. |

- Both filters require `Packet::SIZE` and `Packet2::SIZE` to be 8.
- In the Intra-Slice Chain, both filters require the `Standalone` context entered by `vector_intra_slice_tag()`.
  The preceding stage must be Tag, Logic Cluster, Fxp Cluster, FxpToFp Conversion, Widen, FpToFxp Conversion, or Clip Cluster. See [Intra-Slice Chain stages](./intra-slice-chain.md#stages).
- The Inter-Slice Reducer sets all tag bits to `0`.
  To filter by tag after the reducer, set new tags on its output with `vector_intra_slice_tag()`.

<a id="select-by-tag"></a>

## Filter by tag

A tag is a four-bit value that `vector_intra_slice_tag()` assigns to each input element.
The `TagGuard` passed to `vector_filter_by_tag()` specifies which tag values to keep.
An element whose tag passes the test is a *survivor*.
Output position `k` of a slice holds its `k`-th survivor, not the input element at position `k`.
The output therefore names its positions with new axes, and an output that reuses an axis of the input `Time` or `Packet` is rejected.
The example keeps the nonnegative elements in each slice in three steps.

1. Put `Cmp::Less(0)` first in `TagMode::Comparison` to set tag bit 0 to `1` for negative values and `0` otherwise.
2. With `TagGuard::matches()`, `BitReq::Zero` requires tag bit 0 to be `0`, and the three `BitReq::Ignore` entries accept either value for the other bits.
   Pass that guard to `vector_filter_by_tag()`.
3. Call `vector_final()` to exit the Vector Engine, then `commit_trim::<m![P]>()` to keep all eight positions of each packet for the write.
   Call `commit_view()` to write to an existing tensor in Data Memory (DM).

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![S = 64, K = 32, T4 = 4, P = 8];

fn keep_nonnegative<'l>(
    input: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![S], m![K / 8], m![K % 8]>,
    output: DmTensorViewMut<'l, i32, m![1], m![1], m![S], m![T4, P]>,
) {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False,
        ]))
        .vector_filter_by_tag::<m![T4], m![P]>(TagGuard::matches([
            BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore,
        ]))
        .vector_final()
        .commit_trim::<m![P]>()
        .commit_view(output);
}
```

### Packets emitted and written

The table shows one input slice to `keep_nonnegative()` and its destination slice after `commit_view()`.
Both slices hold four packets of eight elements, and the destination starts with `99` in every position.

| Packet index | Input packet | Destination packet after `commit_view()` |
| --- | --- | --- | --- |
| 0 | `[-2, 4, -1, 7, -3, -4, -5, -6]` | `[4, 7, 8, 9, 10, 11, 12, 13]` |
| 1 | `[8, 9, 10, 11, 12, 13, 14, -1]` | `[14, 0, 0, 0, 0, 0, 0, 0]` |
| 2 and 3 (each) | `[-1, -1, -1, -1, -1, -1, -1, -1]` | `[99, 99, 99, 99, 99, 99, 99, 99]` |

The nine survivors fill packet 0 and the first position of packet 1.
The filter fills the rest of packet 1 with zeros, and `commit_view()` leaves packets 2 and 3 at their initial value of `99`.
See [Writing a tag-filtered stream](../../moving-tensors/commit-engine.md#writing-a-tag-filtered-stream) for an example that initializes the destination first.

For `n` survivors in one slice, the packet counts are as follows.

| Quantity per slice | Packet count |
| --- | --- |
| Emitted by `vector_filter_by_tag()` | \\(\lceil n / 8 \rceil\\) |
| Written by `commit_view()` on hardware | \\(\min(\lceil n / 8 \rceil, \text{Time2::SIZE})\\) |

If `n` is zero, the destination slice keeps all of its previous values.

### Select values using a separate condition

For each position `i`, the filter keeps `values[i]` if `condition[i]` is zero or positive.
The table shows the first two positions of one slice.

| Position `i` | `condition[i]` | Tag bit 0 | `values[i]` | Filter result |
| --- | --- | --- | --- | --- |
| 0 | `-1` | `1` | `7` | Drop `7` |
| 1 | `0` | `0` | `5` | Keep `5` |

1. `vector_intra_slice_tag()` sets tag bit 0 to `1` where `condition[i] < 0` and to `0` elsewhere.
2. `vector_logic(LogicBinaryOpI32::BitAnd, 0)` replaces each stream value with `0`.
   `vector_fxp(FxpBinaryOp::AddFxp, values)` then adds `values[i]` to that zero.
   Both calls preserve the tag from step 1.
3. `vector_filter_by_tag()` keeps the stream values whose tag bit 0 is `0`.

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![S = 64, F = 8, P = 8];

fn select_values<'l>(
    condition: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![S], m![1], m![F]>,
    values: &'l VrfTensor<i32, m![1], m![1], m![S], m![F]>,
    output: DmTensorViewMut<'l, i32, m![1], m![1], m![S], m![P]>,
) {
    condition
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False,
        ]))
        .vector_logic(LogicBinaryOpI32::BitAnd, 0)
        .vector_fxp(FxpBinaryOp::AddFxp, values)
        .vector_filter_by_tag::<m![1], m![P]>(TagGuard::matches([
            BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore,
        ]))
        .vector_final()
        .commit_trim::<m![P]>()
        .commit_view(output);
}
```

### Input and capacity constraints

The tag filter requires every input packet to contain eight elements and rejects padding in either `Time` or `Packet`.
In the example below, `K % 4 # 8` gives each packet four elements followed by four padding positions.

```rust,ignore
axes![S = 64, K = 16, T2 = 2, P = 8];

fn padded_input(
    input: CollectTensor<'_, { Tu::Main }, i32, m![1], m![1], m![S], m![K / 4], m![K % 4 # 8]>,
) {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Rejected because positions 4 through 7 of each input packet are padding.
        .vector_filter_by_tag::<m![T2], m![P]>(TagGuard::all());
}
```

Every packet position carries a tag, including padding, and the guard tests only those tag bits.
Here `TagMode::Zero` would give all eight positions the tag `0`, and `TagGuard::all()` would accept all eight, including the four padding positions.
With `TagMode::Comparison`, a `#` padding position can also pass a guard because its value is unspecified.

`Time2::SIZE * Packet2::SIZE` counts all output positions in a slice.
In the example below, `m![E11 # 16]` declares output elements at positions 0–10 and adds `#` padding at positions 11–15 to complete the second packet.

- Put the output elements first and add only enough padding to complete the last packet.
  For 11 output elements, `m![E11 # 16]` fits in two packets.
  `m![E11 # 24]` is rejected because the third packet would contain only padding.
- Declare at most as many output elements as input elements per slice.
  For example, a filter reading one eight-element packet cannot declare 16 output elements.
- To keep every survivor, the kernel writer must ensure each slice has at most as many survivors as the output declares elements.
  In this example that limit is 11.
  If all input elements could pass the guard, declare as many output elements as there are input elements per slice.
- A slice whose survivors fill more packets than the output holds is an error, unless the commit skips that slice.
  In this example that is a slice with more than 16 survivors.
- On padded chips or clusters, such as `Cluster = m![1 # 2]`, `Time2` must hold every packet a slice reads.
  Their undefined values may pass the guard, and a commit cannot skip a chip or a cluster.

The following kernel reads 24 elements per slice.
The code that prepares the input writes nonnegative values to at most 11 positions per slice and writes `-1` to the rest, so the guard keeps at most 11 elements.

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![S = 64, K24 = 24, E11 = 11];

fn keep_present_candidates<'l>(
    input: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![S], m![K24 / 8], m![K24 % 8]>,
    output: DmTensorViewMut<'l, i32, m![1], m![1], m![S], m![E11 # 16]>,
) {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False,
        ]))
        .vector_filter_by_tag::<m![E11 # 16 / 8], m![E11 # 16 % 8]>(TagGuard::matches([
            BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore,
        ]))
        .vector_final()
        .commit_trim::<m![E11 # 16 % 8]>()
        .commit_view(output);
}
```

When a slice has nine survivors, output positions 0–8 hold those nine values, and the filter fills positions 9 and 10 with zero.
Positions 11–15 are marked `#`, so their values are unspecified whenever the second packet is written.
Initializing the destination in advance does not guarantee a value for those five positions after that write.

Replacing the guard with `TagGuard::all()` makes all 24 input elements survive and emits three packets per slice, one past the output.
The host model and the device simulator reject such a slice.
On the device, a slice that emits four or more packets past its output stops the kernel.

To get zeros after the last survivor, declare all 16 positions as output elements in both the filter output and the destination and initialize the destination with `memset(0, sub)` before `commit_view()`.
For example, if seven elements survive, the filter supplies the zero at position 7 in the first packet, and the second packet keeps the zeros from `memset()`.
See [Writing a tag-filtered stream](../../moving-tensors/commit-engine.md#writing-a-tag-filtered-stream).

### Supported output paths

The tag-filtered stream keeps its per-slice packet counts through the operations before `commit_view()`.

| Goal | Calls |
| --- | --- |
| Write to an existing DM tensor | `vector_final()` → `commit_trim()` → `commit_view()` |
| Convert elements to another type | `vector_cast()` before `vector_final()` |
| Read each element's existing bits as another type | `vector_reinterpret()` before `vector_final()` |
| Convert `f32` elements to `bf16` when the stream uses `Tu::Main` | `commit_cast()` after `commit_trim()` |

To store the result in the vector register file (VRF) or transpose it, first write it to DM with `commit_view()` and use that tensor as input to a new pipeline.
The following direct output calls are unavailable on a `TagFiltered` stream.

```rust,compile_fail
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![S = 64, H = 2, P = 8];
type Filtered<'l> = VectorFinalTensor<
    'l, { Tu::Main }, i32,
    m![1], m![1], m![S], m![H], m![P], CurrentBackend, TagFiltered,
>;

fn fresh_destination(stream: Filtered<'_>) {
    // A fresh DM tensor would have undefined values in unwritten packets.
    stream.commit_trim::<m![P]>().commit::<m![H, P]>();
}

fn register_file(stream: Filtered<'_>, sub: &mut TuContext<{ Tu::Sub }>) {
    // A slice that emits too many packets could write past the VRF destination.
    stream.to_vrf::<m![H, P]>(sub);
}

fn transpose(stream: Filtered<'_>) {
    // The Transpose Engine can discard packets from an incomplete input block.
    stream.transpose::<m![P], m![H # 8]>();
}
```

<a id="select-packet-positions"></a>

## Filter by bitmap

`BITMAP` is an eight-bit mask.
If bit `i` is `1`, `vector_filter_by_bitmap()` keeps position `i` of every input packet.
Bit 0 is the rightmost bit in the binary literal and selects the first position in the packet.
At least one bit must be `1`.

Each of the four input packets below has a row sum at position 0.
`0b0000_0001` keeps those four sums in one output packet, and `commit_trim::<m![N]>()` keeps just the four sums for the DM write.

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![S = 64, N = 4];

fn pack_row_sums<'l>(
    input: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![S], m![N], m![1 # 8]>,
) -> DmTensor<i32, m![1], m![1], m![S], m![N]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_filter_by_bitmap::<0b0000_0001, m![1], m![N # 8]>()
        .vector_final()
        .commit_trim::<m![N]>()
        .commit()
}
```

### Derive the output placement

1. Describe the selected positions of one input packet using the input axes or new axes.
2. Append that placement to the input `Time` placement.
   The number of selected positions in each slice is `input packets × number of 1 bits in BITMAP`.
3. Pad to a multiple of eight and split into eight-element packets.
   Use the placement across packets for `Time2` and the placement within a packet for `Packet2`.

In `pack_row_sums()`, the input `Time` placement is `m![N]` with four rows, and the bitmap keeps one value per row.
The four values and four added padding positions fit in one packet, so the output uses `Time2 = m![1]` and `Packet2 = m![N # 8]`.
The bitmap filter fills the positions added in step 3 with zero.
After `vector_final()`, the result can go to DM through the Commit Engine, to VRF, or to the Transpose Engine.

### Keep the input axes

With an eight-element field axis `F`, the bitmap `0b0000_1111` selects fields 0 through 3, which you express as `F = 4`.
The bitmap `0b0101_0101` selects fields 0, 2, 4, and 6, expressed as `F / 2`.
Field 6 keeps its index along `F` even though it is fourth in the list of selected fields.

The row axis `R` and field axis `F` each have size 8, `C` has size 7, `N` and `X` have size 4, and `Y` has size 2.

| Input `Time`, `Packet` | `BITMAP` | Selected part of one packet | Output `Time2`, `Packet2` |
| --- | --- | --- | --- |
| `m![N / 2]`, `m![N % 2, 1 # 4]` | `0b0001_0001` | `N % 2` | `m![1]`, `m![N # 8]` |
| `m![R]`, `m![C # 8]` | `0b0111_1111` | `C` | `m![[R, C] / 8]`, `m![[R, C] % 8]` |
| `m![R]`, `m![F]` | `0b0000_1111` | `F = 4` | `m![[R, F = 4] / 8]`, `m![[R, F = 4] % 8]` |
| `m![R]`, `m![F]` | `0b0101_0101` | `F / 2` | `m![[R, F / 2] / 8]`, `m![[R, F / 2] % 8]` |
| `m![R]`, `m![X, Y]` | `0b0101_0101` | `X` at coordinate 0 of `Y` | `m![[R, X] / 8]`, `m![[R, X] % 8]` |

### Assign new axes

Use new axes when the input axes cannot express the selected positions or when you want to assign different coordinates.
The new axes must be absent from the input `Time` and `Packet` placements and provide one distinct coordinate per selected position, with no padding of their own.
The whole input `Time` × `Packet` must hold no padding, whichever positions the bitmap selects.
If the bitmap keeps just one position per packet, the input `Time` placement alone can name the selected elements.
For example, with input `m![R]`, `m![F]` from the table, `0b0000_1000` keeps one field per row, so the output uses `m![1]`, `m![R]`.

The following call selects fields 0, 3, and 7 and assigns them coordinates 0, 1, and 2 of a new axis `J`.

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![S = 64, R = 8, F = 8, J = 3];

fn select_fields<'l>(
    input: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![S], m![R], m![F]>,
) -> VectorFilterTensor<'l, { Tu::Main }, i32, m![1], m![1], m![S], m![[R, J] / 8], m![[R, J] % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_filter_by_bitmap::<0b1000_1001, m![[R, J] / 8], m![[R, J] % 8]>()
}
```

### Select padding positions

A bit with value `1` selects its position even when that position holds input padding.

- When keeping the input axes, declare the selected padding with the same padding kind as the input.
- New axes need an input without padding anywhere, so over a padded input the input axes must spell the selected positions.

For input `m![Y]`, `m![1 # 2, X, 1 # 2]` with both axes of size 2, positions 0 and 2 hold elements and the other positions hold padding.
The output placement depends on which padding positions the bitmap selects along with positions 0 and 2.

| `BITMAP` | Selected padding positions per input packet | Output `Time2`, `Packet2` |
| --- | --- | --- |
| `0b0000_0101` | None | `m![1]`, `m![[Y, X] # 8]` |
| `0b0101_0101` | 4 and 6 | `m![1]`, `m![Y, 1 # 2, X]` |
| `0b0000_1111` | 1 and 3 | `m![1]`, `m![Y, X, 1 # 2]` |

### Rejected outputs

The first call assigns the wrong field indices to the selected positions.
The second replaces the input `Time` axes with new axes.

```rust,ignore
axes![S = 64, R = 8, F = 8, K = 16, H = 2, O = 8];

fn wrong_axis(input: CollectTensor<'_, { Tu::Main }, i32, m![1], m![1], m![S], m![R], m![F]>) {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // F = 3 describes fields 0, 1, and 2, while the bitmap selects 0, 3, and 7.
        .vector_filter_by_bitmap::<0b1000_1001, m![[R, F = 3] / 8], m![[R, F = 3] % 8]>();
}

fn renamed_time(
    input: CollectTensor<'_, { Tu::Main }, i32, m![1], m![1], m![S], m![K / 4], m![K % 4 # 8]>,
) {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // The output must preserve K / 4 before packing, as in m![K / 8], m![K % 8].
        .vector_filter_by_bitmap::<0b0000_1111, m![H], m![O]>();
}
```

When the selected positions can be expressed with the input axes, the diagnostic shows a placement that preserves them.
