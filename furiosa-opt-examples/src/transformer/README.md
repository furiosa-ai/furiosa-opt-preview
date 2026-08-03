# qwen3 example

Qwen3-0.6B (28-layer, hidden size 1024) decode-only autoregressive-LM inference, with two aligned pieces:

- `src/kernel/` — the VISA kernels: `attention`, `mlp`, `projection`, `rope`, `rmsnorm`, `residual`, `lm_head`, plus the standalone `final_layer` kernel.
- `src/lib.rs` — the `ops` module: the `#[device]` HBM entry points (`embedding`, `projection`, `attention_forward_first`, `attention_forward`, `decoder`) that `scripts/run.py` launches once per token, per layer.
- `src/axes.rs` — named tensor-dimension constants (`H`, `N`, `D`, `T`, ...) used throughout the kernel type signatures.

The kernels are checked against a plain PyTorch reference, op by op, by `scripts/test.py`.

## Model

```
embedding → { projection → attention → decoder } × 28 layers → final_layer → logits [Wp]
```

Every kernel processes exactly one token — there is no batched prefill stage, so the prompt is fed one token at a time, the same way a generated token is. Each layer runs `projection` (input rmsnorm + q/k/v proj + q/k norm + RoPE) → `attention` (online-softmax over KV-cache chunks `0..=chunk`) → `decoder` (o_proj + residual + MLP + residual).

## Run

### Python (kernel unit tests)

```bash
python3 scripts/test.py
```

Drives each kernel directly with random weights, checked against a PyTorch reference — no model weights required.

### Run the VISA kernels via furiosa-torch (end-to-end generation)

```bash
export RNGD_MODEL_DIR=/path/to/models   # containing Qwen3-0.6B/{model.safetensors,tokenizer.json}
python3 scripts/run.py "The capital of France is" --max-new-tokens 50
```

`--max-new-tokens` (default 300) and `--seed` (default 42) are optional; the prompt is required. Timing (compile, load, warm-up, time-to-first-token, prefill/decode throughput) is printed to stderr.
