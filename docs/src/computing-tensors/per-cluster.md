# Per Cluster

`Device::per_cluster` runs the same body in every cluster on every chip in the device.
The closure reads and writes data for its current chip and cluster, so values can differ between chips and clusters.
The body is an `Fn` closure: it shares no mutable state between chips and clusters.
Tensor results leave through `args`, and the body returns scalar results.

## Interface

```rust,ignore
{{#include ../../../furiosa-opt-std/src/per_cluster.rs:per_cluster}}
```

`args` accepts an SPM argument or a tuple of up to four SPM arguments.
The closure receives a `ClusterContext` and the arguments projected to its current chip and cluster.
`ClusterContext` has no methods yet.
Per-cluster engine control will be reached through it.

| Argument | Closure receives |
| --- | --- |
| `&SpmTensor<D, Chip, Cluster, Pe, Element>` | `SpmTensorLocalRef`, with read access to the current cluster's scratchpad data |
| `&mut SpmTensor<D, Chip, Cluster, Pe, Element>` | `SpmTensorLocalMut`, with read and write access to the current cluster's scratchpad data |
| `&SpmScalar<D, Chip, Cluster>` | `D`, the scalar value for the current chip and cluster |

`SpmScalar` represents one value per active cluster on each chip, as described in [Spatial and Temporal Dimensions](../mapping-tensors/spatial-temporal-dimensions.md#spatial-dimensions).
A plain scalar of type `D` can be captured by the closure and has the same value across all chips and clusters.
Tensor access must go through `args`; HBM and DM tensors are not supported as arguments.

The value the body returns is collected across chips and clusters.
The destination type names the `Chip` and `Cluster` mappings of each collected scalar.

| Body returns | `per_cluster` returns |
| --- | --- |
| `()` | `()` |
| `D` | `SpmScalar<D, Chip, Cluster>` |
| a tuple of up to four `D` values | a tuple of `SpmScalar`s, each with its own `Chip` and `Cluster` |

## Examples

### Element-Wise Computation

The example below subtracts `offset` from each source element and writes the result to a separate tensor.
`source` and `result` each hold four `i32` elements per cluster on PE 0, with the same `Chip` and `Cluster` mappings.
`offset` is a plain scalar captured by the closure.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![C = 2, E = 4];
type Values = SpmTensor<i32, m![1], m![C], m![1 # 4], m![E]>;

fn subtract_into(device: &mut Device, source: &Values, result: &mut Values, offset: i32) {
    device.per_cluster((source, result), |_, (source, mut result)| {
        for i in 0..result.len() {
            result.set(i, source.get(i) - offset);
        }
    });
}
#
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# let source = Values::zero();
# let mut result = Values::zero();
# subtract_into(&mut device, &source, &mut result, 2);
# Ok(())
# }
```

### Per-Cluster Scalar

`SpmTensor::get()` reads the same element position in each active cluster on each chip and returns an `SpmScalar`.
In the example below, `base` has type `SpmScalar<i32, m![1], m![C]>` outside the closure and type `i32` inside it.
The local tensor's `get()` also returns `i32`, so each iteration reads a value before writing the result back to the same element position:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![C = 2, E = 4];
type Values = SpmTensor<i32, m![1], m![C], m![1 # 4], m![E]>;

fn subtract_local_base(device: &mut Device, values: &mut Values) {
    let base: SpmScalar<i32, m![1], m![C]> = values.get(0u32);
    device.per_cluster((&base, values), |_, (base, mut values)| {
        for i in 0..values.len() {
            let current: i32 = values.get(i);
            values.set(i, current - base);
        }
    });
}
#
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# let mut values = Values::zero();
# subtract_local_base(&mut device, &mut values);
# Ok(())
# }
```

### Returned Scalar

The example below returns the sum of two elements from each cluster and writes it back in a second call.
`total` holds one sum per cluster because its `Cluster` mapping is `m![C]`.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![C = 2, E = 4];
type Values = SpmTensor<i32, m![1], m![C], m![1 # 4], m![E]>;

fn sum_into_last(device: &mut Device, values: &mut Values) {
    let total: SpmScalar<i32, m![1], m![C]> =
        device.per_cluster(&*values, |_, values| values.get(1) + values.get(2));
    device.per_cluster((&total, values), |_, (total, mut values)| {
        values.set(3, total);
    });
}
#
# fn main() -> Result<(), Error> {
# let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
# let mut values = Values::zero();
# sum_into_last(&mut device, &mut values);
# Ok(())
# }
```

## Constraints

- **Chip and Cluster**: The host derives the iteration bounds from `Device::num_chip()` and `Device::num_cluster()`, the chip count and the number of clusters per chip.
  It invokes the closure for every chip and cluster within these bounds, including indices marked as padding by an argument's mapping.
  Every argument's `Chip::SIZE` and `Cluster::SIZE` must match the device's chip count and number of clusters per chip.
- **Broadcast outputs**: A `Chip` or `Cluster` mapping that broadcasts declares one value shared by every chip or cluster.
  A body that writes to such an argument, or returns into such a scalar, must produce the same value on each of them.
  The host does not check this.
  `SpmScalar::uniform_value()` checks it on the host when it reads the value.
- **PE placement**: The closure runs on PE 0, so each tensor argument must have exactly one active PE at position 0.
  On a four-PE cluster, use `Pe = m![1 # 4]`.
- **Element indices**: `get()` and `set()` address physical positions within the current cluster's scratchpad data, including padding.
  `len()` returns `Element::SIZE`, and every index must be non-negative and less than `len()`.
  The `Element` mapping contributes only its size: an index is not an axis coordinate.
- **Control flow**: The body uses the same compilation rules as the enclosing device function.
  Element indices can be runtime values, but `for` loops must start at zero and have a positive end known at compile time.
  A branch that executes statements needs a condition with the same value across chips and clusters.
  Selecting between already-computed scalar values, including `min` and `max`, can use a condition that differs between chips and clusters.
  Runtime `panic!` and `assert!` are not yet supported, including the assertion emitted by `clamp`.
- **Engine access**: The closure currently supports scalar scratchpad computations only.
  Issue DMA and Tensor Unit operations outside `per_cluster`, in the enclosing `#[device]` function.
  Per-cluster Tensor Unit control is planned.
