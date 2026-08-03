#!/usr/bin/env python3
"""
Python driver for the Qwen3-0.6B NPU inference pipeline.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import furiosa.torch
from safetensors import safe_open
from tokenizers import Tokenizer

DIR = Path(__file__).parent.parent
CRATE_PATH = DIR
DEVICE = "furiosa"
MODEL_PATH = os.environ["RNGD_MODEL_DIR"]
FUSION = 8

furiosa.torch.set_fusion(FUSION)

# --- axes.rs constants -------------------------------------------------------
H = 1024
C = 64
P = 1024
Q = 2048
W = 151936
Wp = 19 * 8192  # 155648
N = 8
G = 2
D = 128
L = 3072
T = 512
LAYERS = 28

# `dma_scatter` byte-offset row stride
ROW_STRIDE_ND = N * D * 2  # bf16 [.., N, D] row (KV cache)

KERNEL_NAMES = [
    "embedding",
    "projection",
    "attention_forward_first",
    "attention_forward",
    "decoder",
    "final_layer"
]


# --- kernel compilation -------------------------------------------------------
def compile_kernels():
    """Compiles (or loads from cache) all `ops::*` kernels plus `kernel::final_layer`.
    See `cached_compile_from_visa` for the caching mechanism."""
    from furiosa.torch._module.visa import compile_from_visa

    kernels = {}
    for name in KERNEL_NAMES:
        full_name = f"ops::{name}"
        kernels[full_name] = compile_from_visa(CRATE_PATH, full_name).to(DEVICE)
    return kernels


# --- weight loading -----------------------------------------------------------
def load_projection(tensors, layer):
    """Weights consumed by a projection kernel (input rmsnorm + q/k/v proj + q/k norm)."""
    prefix = f"model.layers.{layer}"
    return {
        "input_layernorm": tensors.get_tensor(f"{prefix}.input_layernorm.weight").to(DEVICE),
        "q_proj_w": tensors.get_tensor(f"{prefix}.self_attn.q_proj.weight").to(DEVICE),
        "k_proj_w": tensors.get_tensor(f"{prefix}.self_attn.k_proj.weight").to(DEVICE),
        "v_proj_w": tensors.get_tensor(f"{prefix}.self_attn.v_proj.weight").to(DEVICE),
        "q_proj_n": tensors.get_tensor(f"{prefix}.self_attn.q_norm.weight").to(DEVICE),
        "k_proj_n": tensors.get_tensor(f"{prefix}.self_attn.k_norm.weight").to(DEVICE),
    }


def load_decoder(tensors, layer):
    """Weights consumed by a decoder kernel (output proj + post rmsnorm + MLP)."""
    prefix = f"model.layers.{layer}"
    return {
        "post_attention_layernorm": tensors.get_tensor(f"{prefix}.post_attention_layernorm.weight").to(DEVICE),
        "o_proj_w": tensors.get_tensor(f"{prefix}.self_attn.o_proj.weight").to(DEVICE),
        "mlp_up": tensors.get_tensor(f"{prefix}.mlp.up_proj.weight").to(DEVICE),
        "mlp_gate": tensors.get_tensor(f"{prefix}.mlp.gate_proj.weight").to(DEVICE),
        "mlp_down": tensors.get_tensor(f"{prefix}.mlp.down_proj.weight").to(DEVICE),
    }


def load_model():
    """Loads and uploads every weight, one projection+decoder pair per layer:
    `layers[L]` holds layer `L`'s projection weights and layer `L`'s decoder
    weights, launched as two separate kernels."""
    with safe_open(f"{MODEL_PATH}/Qwen3-0.6B/model.safetensors", framework="pt") as tensors:
        embedding_table = tensors.get_tensor("model.embed_tokens.weight").to(DEVICE)
        final_norm = tensors.get_tensor("model.norm.weight").to(DEVICE)
        lm_head_w = tensors.get_tensor("lm_head.weight").to(DEVICE)

        layers = [
            {"projection": load_projection(tensors, layer), "decoder": load_decoder(tensors, layer)}
            for layer in range(LAYERS)
        ]

    return {
        "embedding_table": embedding_table,
        "final_norm": final_norm,
        "lm_head_w": lm_head_w,
        "layers": layers,
    }


# --- precompute ---------------------------------------------------------------
def rope_chunk(chunk):
    """cos/sin for `chunk`'s absolute positions `chunk*T .. chunk*T+T-1`"""
    half = D // 2
    theta = 1_000_000.0
    inv_freq = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))  # (half,)
    positions = (chunk * T + np.arange(T)).astype(np.float32)  # (T,)
    freqs = np.outer(positions, inv_freq)  # (T, half)

    cos = np.empty((T, D), dtype=np.float32)
    sin = np.empty((T, D), dtype=np.float32)
    cos[:, :half] = np.cos(freqs)
    cos[:, half:] = np.cos(freqs)
    sin[:, :half] = -np.sin(freqs)
    sin[:, half:] = np.sin(freqs)

    cos_t = torch.from_numpy(cos).to(torch.bfloat16).to(DEVICE)
    sin_t = torch.from_numpy(sin).to(torch.bfloat16).to(DEVICE)
    return cos_t, sin_t


def build_masks():
    """The two attention-mask shapes a single-token decode step ever needs:
    `diag` -- `[T, T]` causal mask, since the current (diagonal) KV-cache
    chunk's rows past the query position `row` haven't been written yet, so
    row `r` must only attend to keys `0..=r`. `line` -- `[T]` all-ones mask,
    since an earlier KV-cache chunk is always fully written and every one of
    its `T` keys is attended."""
    return {
        "diag": torch.ones(T, T, dtype=torch.float32).tril(diagonal=0).to(DEVICE),
        "line": torch.ones(T, dtype=torch.float32).to(DEVICE),
    }


def select_mask(masks, chunk, kv_chunk, row):
    return masks["diag"][row] if chunk == kv_chunk else masks["line"]


def build_kv_offset_table():
    """One `dma_scatter` byte-offset per row `0..T`"""
    offsets = np.arange(T, dtype=np.int32) * ROW_STRIDE_ND
    return torch.from_numpy(offsets).to(DEVICE)


class Workspace:
    def __init__(self):
        self.masks = build_masks()
        self.kv_offset_table = build_kv_offset_table()
        self.rope_cache = {}  # chunk -> (cos [T,D], sin [T,D])
        self.kv_cache = {}  # (chunk, layer) -> (k [T,N,D], v [T,N,D])

    def rope(self, chunk):
        if chunk not in self.rope_cache:
            self.rope_cache[chunk] = rope_chunk(chunk)
        return self.rope_cache[chunk]

    def kv(self, chunk, layer):
        key = (chunk, layer)
        if key not in self.kv_cache:
            k = torch.zeros(T, N, D, dtype=torch.bfloat16).to(DEVICE)
            v = torch.zeros(T, N, D, dtype=torch.bfloat16).to(DEVICE)
            self.kv_cache[key] = (k, v)
        return self.kv_cache[key]

    def warm(self, chunks):
        """Forces the lazy KV-cache/RoPE allocation for every chunk index this
        run will ever touch, up front. Without this, the *first* kernel launch
        that reaches a new chunk pays its `.to(DEVICE)` allocation cost."""
        for chunk in chunks:
            self.rope(chunk)
            for layer in range(LAYERS):
                self.kv(chunk, layer)


# --- token step ----------------------------------------------------------------
def run_kernel(kernels, model, ws, scratch, token, pos, need_logits):
    """Runs a single token (a real prompt token or a previously-sampled token)
    through the model at absolute (0-indexed) position `pos`, updating the KV
    cache."""
    x, q, attn_max, attn_sum, attn_out, logits = scratch

    chunk = pos // T
    row = pos % T

    # `chunk` (and so the RoPE table and kv-offset row) is the same for every
    # layer within one token step -- computed once here instead of
    # redundantly by every `projection` call below (28 calls/step otherwise).
    cos_full, sin_full = ws.rope(chunk)
    cos_row, sin_row = cos_full[row], sin_full[row]
    kv_offset_row = ws.kv_offset_table[row : row + 1]

    # Device-indexed rows everywhere except the KV-cache scatter, which still
    # goes through the real `dma_scatter`-based kernel with a sliced offset.
    kernels["ops::embedding"](model["embedding_table"][token], x)

    def projection(proj, layer):
        k_cache, v_cache = ws.kv(chunk, layer)
        kernels["ops::projection"](
            x,
            proj["q_proj_w"],
            proj["k_proj_w"],
            proj["v_proj_w"],
            proj["input_layernorm"],
            proj["q_proj_n"],
            proj["k_proj_n"],
            kv_offset_row,
            cos_row,
            sin_row,
            k_cache,
            v_cache,
            q,
        )

    def attention(layer):
        for kv_chunk in range(chunk + 1):
            mask = select_mask(ws.masks, chunk, kv_chunk, row)
            k_cache, v_cache = ws.kv(kv_chunk, layer)
            args = (q, k_cache, v_cache, mask, attn_max, attn_sum, attn_out)
            if kv_chunk == 0:
                kernels["ops::attention_forward_first"](*args)
            else:
                kernels["ops::attention_forward"](*args)

    def decoder(layer_weights):
        kernels["ops::decoder"](
            attn_out,
            attn_sum,
            x,
            layer_weights["o_proj_w"],
            layer_weights["post_attention_layernorm"],
            layer_weights["mlp_up"],
            layer_weights["mlp_gate"],
            layer_weights["mlp_down"],
        )

    for layer, layer_weights in enumerate(model["layers"]):
        projection(layer_weights["projection"], layer)
        attention(layer)
        decoder(layer_weights["decoder"])

    if need_logits:
        kernels["ops::final_layer"](x, model["final_norm"], model["lm_head_w"], logits)


# --- sampling -----------------------------------------------------------------
def sample_token(logits_dev):
    return int(logits_dev[:W].argmax().cpu())


def build_scratch():
    """Activation scratch for a token step: x, q, attn_max, attn_sum,
    attn_out, logits. Allocated once during setup, before timing starts."""
    return (
        torch.zeros(H, dtype=torch.bfloat16).to(DEVICE),
        torch.zeros(N, G, D, dtype=torch.bfloat16).to(DEVICE),
        torch.zeros(N, G, dtype=torch.float32).to(DEVICE),
        torch.zeros(N, G, dtype=torch.float32).to(DEVICE),
        torch.zeros(N, G, D, dtype=torch.bfloat16).to(DEVICE),
        torch.zeros(Wp, dtype=torch.bfloat16).to(DEVICE),
    )


# --- main ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B NPU inference pipeline.")
    parser.add_argument("prompt", help="Prompt to complete.")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Number of tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    t0 = time.time()
    kernels = compile_kernels()
    compile_time = time.time() - t0

    t0 = time.time()
    model = load_model()
    load_time = time.time() - t0

    tokenizer = Tokenizer.from_file(f"{MODEL_PATH}/Qwen3-0.6B/tokenizer.json")
    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    num_tokens = len(ids)

    ws = Workspace()
    scratch = build_scratch()

    t0 = time.time()
    max_chunk = (num_tokens + args.max_new_tokens - 1) // T
    ws.warm(range(max_chunk + 1))
    sample_token(scratch[-1])
    warmup_time = time.time() - t0

    print(args.prompt, end="", flush=True)

    # Every prompt token is run through the model one at a time, the same way
    # a generated token is -- there is no batched prefill stage. 
    token = None
    prefill_time = None
    run_start = time.time()
    with torch.no_grad():
        for pos in range(num_tokens + args.max_new_tokens):
            cur_token = ids[pos] if pos < num_tokens else token
            need_logits = pos >= num_tokens - 1
            run_kernel(kernels, model, ws, scratch, cur_token, pos, need_logits)
            if need_logits:
                token = sample_token(scratch[-1])
                if prefill_time is None:
                    prefill_time = time.time() - run_start
                text = tokenizer.decode([token], skip_special_tokens=False)
                print(text, end="", flush=True)
    run_time = time.time() - run_start
    print()

    decode_time = run_time - prefill_time
    prefill_tps = num_tokens / prefill_time
    decode_tps = args.max_new_tokens / decode_time

    print(f"kernel compile time: {compile_time:.2f}s", file=sys.stderr)
    print(f"model load time:     {load_time:.2f}s", file=sys.stderr)
    print(f"workspace warm-up:   {warmup_time:.2f}s", file=sys.stderr)
    print(f"time to first token: {prefill_time * 1000:.2f}ms", file=sys.stderr)
    print(f"prefill:             {prefill_tps:.1f} tok/s ({num_tokens} tokens)", file=sys.stderr)
    print(f"decode:              {decode_tps:.1f} tok/s ({args.max_new_tokens} tokens)", file=sys.stderr)


if __name__ == "__main__":
    main()
