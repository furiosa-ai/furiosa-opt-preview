# Cast Compaction

The Vector Engine's output cast-compaction stage converts `i32` and `f32` results to lower-precision types.

## Interface

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:cast_impl}}
```

`vector_cast()` is a Vector Engine stage between `vector_init()` and `vector_final()`.

`.vector_cast::<OutD, OutPacket>()` converts each element to type `OutD` and pads the output back to one 32-byte flit.

Cast compaction changes element precision while preserving the tensor’s logical shape; it is not a [mathematical tensor move](../../mapping-tensors/tensor-semantics.md#mathematical-tensor-move).
All dimensions pass through unchanged, except for the `Packet` layout, which repads so the output still fits one 32-byte flit.

The example below casts an 8-element `i32` packet (8 × 4 = 32 bytes) to `i8`.
After the cast, the 8 elements occupy 8 bytes, so `A # 32` pads the output back to 32 bytes:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 4, A = 8];

fn cast_i32_to_i8<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![B], m![A]>,
) -> VectorCastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![B], m![A # 32]> {
    input.vector_init().vector_cast()
}
# 
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# 
# let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![B], m![A]> = CollectTensor::new(&mut device.main, Tensor::zero());
# let _o = cast_i32_to_i8(c);
#
# Ok(())
# }
```

The input data may not fill 32 bytes.
The example below casts an `i32` input where 4 data elements are padded to 8 (`A # 8`, 32 bytes) into an `i8` output where the same 4 elements are padded to 32 (`A # 32`, also 32 bytes):

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 4];

fn cast_padded<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 8]>,
) -> VectorCastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 32]> {
    input.vector_init().vector_cast()
}
# 
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# 
# let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
# let _o = cast_padded(c);
#
# Ok(())
# }
```

## Supported Casts

Each input is a 32-byte flit.
The supported source types are `f32` and `i32`, each with specific target types:

| Input Type (`D`) | Supported Output Types (`OutD`) |
| --- | --- |
| `i32` | `i4`, `i8`, `i16` |
| `f32` | `f8e5m2`, `f8e4m3`, `bf16` |

## Performance

Cast compaction processes one flit per cycle regardless of how much of the flit carries valid data.
The downstream [Commit Engine](../../moving-tensors/commit-engine.md) aggregates under-utilized flits into dense DM writes, so no DM bandwidth is wasted.
