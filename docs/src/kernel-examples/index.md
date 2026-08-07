# Kernel Examples

Use this chapter to choose a pattern that combines mapping, movement, computation, and scheduling.

## Choose a Starting Point

Start implementation from a [Quick Start](../quick-start.md) pattern whenever it can express the contract.
The remaining pages are design references until their code is verified against the current API.

| Need | Start with | Technical focus and status |
|------|------------|----------------------------|
| Elementwise operation or basic contraction | [Quick Start](../quick-start.md) | Base-template source examples are generated and tested in CI with emulation. Verify the target release before reuse. |
| Tiling and accumulation strategy | [Tiling](./tiling.md) | Tile size, memory layout, and accumulation strategy. Conceptual guidance. |
| Layout transformation through fetch and commit | [Fetch and Commit Engine](./fetch-commit-engine.md) | Axis permutation, full-flit writes, packet padding, and explicit tiled segmentation. Conceptual guidance rather than copyable source. |
| Reduction split across tensor instances | [Split Reduce](./split-reduce.md) | Interleaved Fetch reduces across tensor instances when one tile cannot hold the reduction. Conceptual guidance rather than copyable source. |
| Reduction across chips or clusters | [Chip/Cluster Reduce](./chip-cluster-reduce.md) | DMA broadcast with Vector Engine binary add for ReduceScatter and AllReduce patterns. Conceptual guidance. |
| Transformer or MoE architecture | [Transformer](./transformer.md) or [Mixture of Experts](./mixture-of-experts.md) | Tiling, multi-chip reduction, memory management, and sparse dynamic routing. Architecture guidance, not vISA implementation recipes. |
| Double-buffered GEMM | [Tiling](./tiling.md) and [Fetch and Commit Engine](./fetch-commit-engine.md) | HBM DMA, sub-context TRF prefetch, main-context tiled contraction, cast, and commit. No complete example is available yet. |

## Place and Reduce Dimensions

The table below summarizes the available parallelism and reduction at each level:

| Dimension | Type | Defined in | Reduced in |
|-----------|------|------------|------------|
| `Chip` | Spatial | [HBM, SRAM](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram), [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Cluster` | Spatial | [SRAM](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram), [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Slice` | Spatial | [SRAM](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram), [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [Vector](../computing-tensors/vector-engine/index.md) |
| `Lane` | Spatial | [TRF](../mapping-tensors/spatial-temporal-dimensions.md#hbm-and-sram) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Time` | Temporal | [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Packet` | Spatial | [Stream](../mapping-tensors/spatial-temporal-dimensions.md#tensor-unit-stream) | [Contraction](../computing-tensors/contraction-engine/index.md) |

The `Chip` and `Cluster` rows above involve cross-chip and cross-cluster reduction patterns.
See [Chip/Cluster Reduce](./chip-cluster-reduce.md), which demonstrates DMA broadcast followed by Vector Engine binary add.
