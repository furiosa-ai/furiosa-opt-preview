# Qwen3 example

Qwen3-0.6B (28 layers, hidden size 1024) decode-only inference:

- `kernel/` + `mod.rs` — the VISA kernels, and the six `#[device]` entry points a token step launches (`embedding`, `projection`, `attention_forward_first`, `attention_forward`, `decoder`, `final_layer`). `axes.rs` holds the extents their type signatures speak.
- `../../tests/transformer_tests.rs` — smoke tests for every entry point, and scalar-oracle tests for projection, chunked attention, decoder, final layer, and one connected decoder step.

## Model

```
embedding → { projection → attention → decoder } × 28 layers → final_layer → logits [Wp]
```

Every kernel processes exactly one token, so a prompt token is fed the same way a generated token is — there is no batched prefill stage. Each layer runs `projection` (input rmsnorm + q/k/v proj + q/k norm + RoPE) → `attention` (online softmax over KV-cache chunks `0..=chunk`) → `decoder` (o_proj + residual + MLP + residual).

## Run

```bash
cargo furiosa-opt test --test transformer_tests -- --nocapture
```
