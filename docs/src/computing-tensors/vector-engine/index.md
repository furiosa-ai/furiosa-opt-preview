# Vector Engine

The Vector Engine performs elementwise computation, reduction, filtering, and output casts.
Examples include activations (GELU, SiLU), normalizations (softmax, layer norm), binary operations, and intra- and inter-slice reductions.

The engine accepts only 32-bit types, `i32` and `f32`.
An upstream [Contraction Engine](../contraction-engine/index.md) widens types automatically (`bf16` products accumulate in `f32`, `i8` products in `i32`).
When that engine is bypassed, the [Fetch Adapter](../fetch-adapter.md#type-casting) must widen the input via its type-cast adapter.

## Interface

In a single Tensor Unit invocation, the Vector Engine portion is the method chain from `vector_init()` to `vector_final()`.
The computation stages have two parts: the [Intra-Slice Chain](./intra-slice-chain.md) (elementwise / binary / per-slice reduce stages) and the [Inter-Slice Reducer](./inter-slice-reducer.md) (reduces across the 256 slices in a cluster).
Between `vector_init()` and `vector_final()`, the chain alone, the reducer alone, or both can run, followed by an optional filter and cast.
When both run, the order is `IntraFirst` (chain then reducer) or `InterFirst` (reducer then chain).
Both computation blocks can be skipped when only an output cast is needed.

The intra-slice chain is entered via `vector_intra_slice_tag()`, or `vector_intra_slice_unzip()` when the input carries a 2-way grouping axis to split into two parallel streams (see [Pair Mode](./intra-slice-chain.md#pair-mode)).
Either entry point fires right after `vector_init()` or on the inter-slice reducer's output.
The inter-slice reducer is entered via `vector_inter_slice_reduce()`, either right after `vector_init()` or from a [compatible intra-slice stage](./intra-slice-chain.md#transitioning-to-the-inter-slice-reducer).
For stage-by-stage API coverage, see [Intra-Slice Chain](./intra-slice-chain.md) and [Inter-Slice Reducer](./inter-slice-reducer.md).

For the output-stage APIs, see [Filter](./filter.md) and [Cast](./cast.md).

The signatures below cover the `vector_init()`-side entry methods only.
The same method names (`vector_intra_slice_tag`, `vector_inter_slice_reduce`) also exist on chain and reducer tensors for the chain↔reducer transitions; those are documented in the child pages.

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/mod.rs:vector_init_impl}}

{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_init_intra_slice_methods_impl}}

{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_init_inter_slice_reduce_i32_impl}}
```

## Output stages

The Intra-Slice Chain and Inter-Slice Reducer are the computation blocks.
Filter and Cast are output stages: they sit outside both computation blocks and can follow either one.
Where a component sits in the engine and what its operations span are different properties:

| Component | Role | Operation scope |
| --- | --- | --- |
| [Intra-Slice Chain](./intra-slice-chain.md) | Computation block | Elementwise operations and reductions within each slice |
| [Inter-Slice Reducer](./inter-slice-reducer.md) | Computation block | Reduction across slices |
| [Filter](./filter.md) | Output stage | Selection and packing independently within each slice |
| [Cast](./cast.md) | Output stage | Conversion of each emitted packet |

A computation block does not have to run every stage before entering an output stage.
The method chain selects a compatible exit point and skips the remaining computation stages.
The supported output paths are:

| Current position | Available next steps |
| --- | --- |
| Immediately after `vector_init()` | Cast |
| Compatible 8-way intra-slice stage | Filter, Cast, or `vector_final()` |
| Inter-Slice Reducer output | Filter, Cast, or `vector_final()` |
| Filter output | Cast or `vector_final()` |
| Cast output | `vector_final()` |

Filter requires the `Standalone` context; the ordinary cast and finalization paths also accept the merged result of pair mode.
Once a stream enters Filter or Cast, it cannot return to either computation block.

When both blocks run, the output stages read the result of the second block, whichever order the two ran in.
The two functions below run the blocks in opposite orders and end with the same filter and cast.
Each sums over `B` across slices, keeps the even packet positions, and casts them to `i16`:

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 8, F = 8, E4 = 4];

// IntraFirst: ReLU in each slice, then the sum across slices.
fn chain_then_reducer<'l>(
    input: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![A, B], m![1], m![F]>,
) -> VectorFinalTensor<'l, { Tu::Main }, i16, m![1], m![1], m![A, 1 # 8], m![1], m![E4 # 16]> {
    input.vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpI32::Max, 0)
        .vector_inter_slice_reduce::<m![A, 1 # 8], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_filter_by_bitmap::<0b0101_0101, m![1], m![E4 # 8]>()
        .vector_cast::<i16, m![E4 # 16]>()
        .vector_final()
}

// InterFirst: the sum across slices, then a bias in each slice.
fn reducer_then_chain<'l>(
    input: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![A, B], m![1], m![F]>,
) -> VectorFinalTensor<'l, { Tu::Main }, i16, m![1], m![1], m![A, 1 # 8], m![1], m![E4 # 16]> {
    input.vector_init()
        .vector_inter_slice_reduce::<m![A, 1 # 8], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_filter_by_bitmap::<0b0101_0101, m![1], m![E4 # 8]>()
        .vector_cast::<i16, m![E4 # 16]>()
        .vector_final()
}
```

A computation block can also be cut short.
Here filtering follows Logic without running the later intra-slice stages:

```rust,no_run
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![S = 64, F = 8, J = 2];

fn filter_after_logic<'l>(
    input: CollectTensor<'l, { Tu::Main }, i32, m![1], m![1], m![S], m![1], m![F]>,
) -> VectorFinalTensor<'l, { Tu::Main }, i8, m![1], m![1], m![S], m![1], m![J # 32]> {
    input.vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f)
        .vector_filter_by_bitmap::<0b0000_1001, m![1], m![J # 8]>()
        .vector_cast::<i8, m![J # 32]>().vector_final()
}
```

## Examples

A few representative examples follow to give a feel for what a Vector Engine call looks like, with the full API tour deferred to the child pages [Intra-Slice Chain](./intra-slice-chain.md) and [Inter-Slice Reducer](./inter-slice-reducer.md).

### ReLU Activation

This pass applies ReLU elementwise, computing \\(output[b, k, m, n] = \max(input[b, k, m, n], 0)\\).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, K = 256, M = 16, N = 16];

// ReLU activation after batched matrix multiplication.
// Chain-only pass, so the reducer is skipped and the path trivially resolves to IntraFirst.
// Both clusters (B = 2) and all 256 slices (K) carry real data, no padding.
fn relu<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // max(x, 0), the ReLU itself
        .vector_clip(ClipBinaryOpF32::Max, 0.0f32)
        .vector_final()
}
# 
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# 
# let c: ContractTensor<'_, _, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]> = ContractTensor::new(&mut device.main, Tensor::zero());
# let _o = relu(c);
#
# Ok(())
# }
```

### ReLU Then Reduce

This pass applies ReLU per slice and then reduces across `R`, giving \\(output[a, b] = \sum_{r \in R} \max(input[a, b, r], 0)\\).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// Chain applies ReLU, then reducer sums across slices.
// IntraFirst shape (chain runs first, then reducer).
// Both clusters (B = 2), all 256 slices (A / 8 * R), and full Way8 packet (A % 8) carry real data.
fn relu_then_reduce<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // max(x, 0), the ReLU
        .vector_clip(ClipBinaryOpI32::Max, 0)
        // sum across R slices
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
# 
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
# let _o = relu_then_reduce(c);
#
# Ok(())
# }
```

### Reduce Then Bias

This pass reduces across `R` and then adds a constant bias, giving \\(output[a, b] = \left(\sum_{r \in R} input[a, b, r]\right) + 100\\).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// Reducer sums across slices, then chain adds a bias to the reduced result.
// InterFirst shape (reducer runs first, then chain).
// Both clusters (B = 2), all 256 slices, and full Way8 packet carry real data.
fn reduce_then_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        // sum across R slices
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_intra_slice_tag(TagMode::Zero)
        // add bias 100
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
}
# 
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
# let _o = reduce_then_add(c);
#
# Ok(())
# }
```

### Intra- and Inter-Slice Reduction

This pass reduces `R` entirely by combining the intra-slice reducer (over `R`'s Time and Packet portions) and the inter-slice reducer (over `R`'s Slice portion).
The einsum form is `BR -> B`, with saturating addition.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, R = 8192];

// R splits across Slice (R / 32 = 256), Time (R % 32 / 4 = 8), and Packet (R % 4, padded to 8 in Way8).
// Chain runs intra-slice reduce over R's Time and Packet portions, then the reducer collapses the Slice portion.
// IntraFirst shape (chain runs first, then reducer).
fn full_sum<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![R / 32], m![R % 32 / 4], m![R % 4 # 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![1 # 256], m![1], m![1 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Way8 → Way4 (back 4 packet positions were padding)
        .vector_narrow_trim::<m![R % 4]>()
        // sum over R's Time and Packet portions
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        // Way4 → Way8
        .vector_widen_pad::<m![1 # 8]>()
        // sum over R's Slice portion across all 256 slices
        .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
# 
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![R / 32], m![R % 32 / 4], m![R % 4 # 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
# let _o = full_sum(c);
#
# Ok(())
# }
```

### Pair Add

This pass unzips two interleaved groups along `I` and adds them pair-wise.
The einsum form is `ABI -> AB`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, I = 2];

// Pair-mode entry via unzip, then a zip op fuses the two streams with an add.
// Both clusters (B = 2), all 256 slices (A / 8), and full Way8 packet (A % 8) carry real data.
fn pair_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        // split into group 0 and group 1 along I
        .vector_intra_slice_unzip::<I, m![1]>()
        // group0 + group1
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
}
# 
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# 
# let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
# let _o = pair_add(c);
#
# Ok(())
# }
```
