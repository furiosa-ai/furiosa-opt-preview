# Qwen3 Decoder Capstone

This capstone traces one Qwen3 decoder token from a host-prepared embedding through the current `furiosa-opt-examples` kernels to padded vocabulary logits, then verifies each boundary with a portable host oracle and a reproducible schedule artifact.

The example is **decode-only and single-token**. It does not implement prefill, sampling, or device-side token lookup. The executable source is the [transformer example](https://github.com/furiosa-ai/furiosa-opt/tree/main/furiosa-opt-examples/src/transformer); this page explains its decisions and points to the source and tests instead of copying the implementation.

## Workload semantics

The workload is one decoder step:

```text
host-prepared embedding → 28 decoder layers → final RMSNorm and LM head → logits [Wp]
```

The current example describes Qwen3-0.6B with 28 layers and hidden size 1024. Each decoder layer has these boundaries:

1. `projection`: input RMSNorm, Q/K/V projections, Q/K RMSNorm, RoPE, and KV-cache writes;
2. `attention_forward_first` or `attention_forward`: one online-softmax attention chunk;
3. `decoder`: output projection, residual, post-attention RMSNorm, MLP, and residual.

`final_layer` applies the final RMSNorm and computes logits into the padded vocabulary shape. The exact axis sizes and tensor types are defined in [`axes.rs`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/axes.rs).

## Decision trace

The capstone follows the same order as the rest of the book:

1. **Workload semantics**: one token and a growing KV cache define the dataflow.
2. **Hard constraints**: every kernel has a typed public boundary; the vocabulary output uses `Wp`, while the model vocabulary is `W`.
3. **Kernel boundaries**: the Rust module exposes `embedding`, `projection`, `attention_forward_first`, `attention_forward`, `decoder`, and `final_layer` as separate device entry points.
4. **Mapping and movement**: the source moves host-prepared tensors through HBM and DM and uses the representations required by each engine. See [Mapping Tensors](../mapping-tensors/index.md) and [Moving Tensors](../moving-tensors/index.md) for those contracts.
5. **Engine choices**: projection and LM-head contractions use the Contraction Engine; RMSNorm and elementwise stages use the Vector Engine; transfers use the DMA and Fetch/Commit paths. See [Computing Tensors](../computing-tensors/index.md) for the contracts.
6. **Evidence**: host-oracle tests establish values, and a dumped schedule establishes the static execution plan. See [Scheduling and Tuning](../scheduling/index.md).

The capstone therefore demonstrates how choices compose; it does not redefine any individual engine API.

## Host-prepared embedding

The `embedding` entry point is a host-prepared HBM copy. It accepts an `HbmTensor<bf16, ..., m![H]>` and copies its view into the output HBM tensor with `to_hbm_view`. It is not a device lookup from token IDs and must not be described as one.

The exact implementation is [`ops::embedding`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/mod.rs), and its smoke test is [`test_embedding`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/tests/transformer_tests.rs).

## Decoder step

`projection` receives one hidden vector, the projection and RMSNorm weights, the position-dependent `cos` and `sin` vectors, and mutable K/V caches. It writes Q to HBM and updates the cache. The first attention chunk uses `attention_forward_first`; later chunks use `attention_forward` to continue the online reduction state.

`decoder` consumes the attention output and its reduction state, writes the residual stream back to HBM, and applies the output projection, residual path, RMSNorm, MLP, and final residual path. The source signatures are the contract; the [projection](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/kernel/projection.rs), [attention](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/kernel/attention.rs), and [decoder](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/src/transformer/kernel/mlp.rs) modules own implementation detail.

The boundary test [`test_decoder_step_wires_kernel_boundaries`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/tests/transformer_tests.rs) connects host-prepared input, projection, one attention chunk, and decoder with zero weights. Its scalar-oracle tests separately cover projection, chunked attention, and decoder arithmetic.

## Final layer and padded logits

`final_layer` takes the residual stream, applies RMSNorm, reshapes the model vocabulary weight view to the padded output representation, and writes logits to `m![Wp]`. The source accepts a separate `lm_head_weight`; this example makes no tied-embedding claim.

The deterministic test [`test_final_layer_matches_scalar_reference`](https://github.com/furiosa-ai/furiosa-opt/blob/main/furiosa-opt-examples/tests/transformer_tests.rs) compares the valid `W` logits with a host RMSNorm-and-dot-product oracle. Padding is part of the tensor contract; interpretation of padded entries belongs to the caller.

## Verify and inspect

Run the transformer test target with the repository's normal command:

```bash
cargo furiosa-opt test --test transformer_tests
```

The tests use seeded small fixtures and compare each supported boundary against host equations. They do not require model files or a sampling loop. For schedule evidence, compile a selected device entry point with `--dump-schedule` and inspect it with the [Schedule Viewer](../appendix/schedule-viewer.md), following the [Scheduling and Tuning](../scheduling/index.md) comparison protocol.

Keep the capstone's changes disciplined: establish the oracle first, record a baseline schedule, change one mapping, movement, engine, or shape choice, and compare the same artifact. Do not infer throughput from the oracle or from source order.
