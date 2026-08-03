#!/usr/bin/env python3
"""Unit tests for the single-token NPU kernels.

Each test drives one `ops::*` kernel with random weights/inputs and compares
against a plain PyTorch reference. Random weights are fine - we are checking
that the kernel computes the *right function*, not the real model.

Run from the crate root (the directory with Cargo.toml and src/), e.g.
    python3 scripts/test.py
It needs the furiosa toolchain; it does NOT need the model weights.
"""
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import furiosa.torch
from furiosa.torch._module.visa import compile_from_visa

SEED = 42
DIR = Path(__file__).parent.parent
CRATE_PATH = DIR
DEVICE = "furiosa"

torch.manual_seed(SEED)
furiosa.torch.set_fusion(8)


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 6.25e-8
    attention_bias: bool = False


# --- reference layers ---------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def precompute_rope(head_dim, max_pos, theta, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_pos).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def neg_half(x):
    # The Rust rope kernel does `rotate_half` as a plain half-swap (no negation)
    # and folds the sign into `sin` (first half negated). Match that on the host.
    half = x.shape[-1] // 2
    return torch.cat((-x[..., :half], x[..., half:]), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=cfg.attention_bias)

    def projection(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        return q, k, v


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Decoder(nn.Module):
    """Second half of a transformer layer: o_proj + residual + post-norm + MLP."""

    def __init__(self, cfg):
        super().__init__()
        self.mlp = MLP(cfg)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, rx):
        x = rx + self.o_proj(x)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# --- helpers -----------------------------------------------------------------
def npu(kernel_name):
    return compile_from_visa(CRATE_PATH, kernel_name).to(DEVICE)


def sample(*shape):
    return torch.rand(*shape, dtype=torch.bfloat16, requires_grad=False)


def ones(*shape):
    return torch.ones(*shape, dtype=torch.bfloat16, requires_grad=False)


BF16_EPS = 2 ** -8  # ~half a bf16 ULP at magnitude 1; the absolute floor for round-off


def report(name, expected, actual):
    # Δ = expected - actual (the per-element error of the NPU kernel's output
    # against the PyTorch reference), always reported as |Δ|.
    e = expected.detach().cpu().float().flatten()
    a = actual.detach().cpu().float().flatten()
    diff = (e - a).abs()

    # Per-element tolerance: 1% of the expected value, or bf16 round-off,
    # whichever is larger. The 1% relative bound dominates almost everywhere;
    # the bf16 floor keeps near-zero expected values from getting an
    # unreasonably tight (near-zero) tolerance.
    tol = torch.maximum(0.01 * e.abs(), torch.full_like(e, BF16_EPS))
    within_tol = diff <= tol
    pass_rate = within_tol.float().mean().item() * 100
    ok = pass_rate >= 99.5

    i = diff.argmax()
    max_diff = diff[i].item()
    max_pct = (diff[i] / e[i].abs().clamp_min(1e-12) * 100).item()
    print(f"[{name:26}] max|Δ|={max_diff:8.4f} ({max_pct:7.2f}% of expected)  "
          f"mean|Δ|={diff.mean().item():8.5f}  elems within tol={pass_rate:6.2f}%  -> {'PASS' if ok else 'FAIL'}")
    return ok


# --- kernel tests ---------------------------------------------------------------
def test_projection(cfg, pos=5):
    """ops::projection: input-norm + q/k/v proj + q/k norm + RoPE, scattering
    K/V into row `pos` of a full `[T, N, D]` KV cache."""
    T = 512  # KV-cache chunk length (axes.rs T)

    cos, sin = precompute_rope(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, torch.bfloat16)
    cos_p, sin_p = cos[pos : pos + 1], sin[pos : pos + 1]  # single position (1, D)

    attn = Attention(cfg).eval().bfloat16()
    rms = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).eval().bfloat16()
    kernel = npu("ops::projection")

    x = sample(1, 1, cfg.hidden_size)  # single token (B, T=1, H)
    with torch.no_grad():
        q, k, v = attn.projection(rms(x), cos_p, sin_p)
    q_cpu = q.transpose(1, 2).reshape(cfg.num_attention_heads, cfg.head_dim)  # (16, D)
    k_cpu = k.transpose(1, 2).reshape(cfg.num_key_value_heads, cfg.head_dim)  # (8, D)
    v_cpu = v.transpose(1, 2).reshape(cfg.num_key_value_heads, cfg.head_dim)  # (8, D)

    sin_p = neg_half(sin_p)

    N, G, D = cfg.num_key_value_heads, 2, cfg.head_dim
    q_npu = sample(N, G, D).to(DEVICE)  # q_out    m![N, G, D]
    k_npu = sample(T, N, D).to(DEVICE)  # k_cache  m![T, N, D]
    v_npu = sample(T, N, D).to(DEVICE)  # v_cache  m![T, N, D]

    # dma_scatter indices are byte offsets along the scattered axis: row `pos` of
    # a `[.., N, D]` bf16 table is `pos * N * D * 2` bytes in.
    index_nd = torch.tensor([pos * N * D * 2], dtype=torch.int32).to(DEVICE)  # indexnd m![1], into [T, N, D]

    kernel(
        x[0, 0].to(DEVICE),                 # x                m![H]
        attn.q_proj.weight.to(DEVICE),
        attn.k_proj.weight.to(DEVICE),
        attn.v_proj.weight.to(DEVICE),
        rms.weight.to(DEVICE),              # input_rms_weight
        attn.q_norm.weight.to(DEVICE),
        attn.k_norm.weight.to(DEVICE),
        index_nd,                           # indexnd
        cos_p[0].to(DEVICE),                # cos              m![D]
        sin_p[0].to(DEVICE),                # sin              m![D]
        k_npu, v_npu, q_npu,                # outputs (k_npu/v_npu are the full cache)
    )

    ok = True
    ok &= report("projection q", q_cpu, q_npu.cpu().view(cfg.num_attention_heads, cfg.head_dim))
    ok &= report("projection k", k_cpu, k_npu.cpu()[pos])
    ok &= report("projection v", v_cpu, v_npu.cpu()[pos])
    return ok


def test_attention(cfg, valid=6):
    """ops::attention_forward_first: single-query attention over one T-block. Exercises the qk and weighted-sum axis gathers.

    forward_first writes the UNNORMALIZED output + running softmax sum; the
    softmax normalization is folded into the decoder kernel, so we divide the
    output by `sum` before comparing against reference softmax attention."""
    N, G, D, T = cfg.num_key_value_heads, 2, cfg.head_dim, 512
    n_heads = cfg.num_attention_heads

    q = sample(n_heads, D)   # (16, D)
    k = sample(T, N, D)      # kv cache  (T, N, D)
    v = sample(T, N, D)
    mask = torch.zeros(T, dtype=torch.float32)
    mask[:valid] = 1.0       # attend the first `valid` keys (1 = attend, 0 = mask)

    with torch.no_grad():
        qf = q.float().view(N, G, D)
        scores = torch.einsum("ngd,tnd->ngt", qf, k.float()) / (D ** 0.5)  # (N, G, T)
        scores = scores.masked_fill(mask.view(1, 1, T) == 0, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out_cpu = torch.einsum("ngt,tnd->ngd", w, v.float())  # (N, G, D)

    max_npu = sample(N, G).float().to(DEVICE)
    sum_npu = sample(N, G).float().to(DEVICE)
    out_npu = sample(N, G, D).to(DEVICE)

    kernel = npu("ops::attention_forward_first")
    kernel(
        q.view(N, G, D).to(DEVICE),  # q     m![N, G, D]
        k.to(DEVICE),                # k     m![T, N, D]
        v.to(DEVICE),                # v     m![T, N, D]
        mask.to(DEVICE),             # mask  m![T]
        max_npu, sum_npu, out_npu,   # outputs
    )

    out_norm = out_npu.cpu().float() / sum_npu.cpu().float().unsqueeze(-1)  # normalize
    return report("attention out", out_cpu, out_norm.view(N, G, D))


def test_decoder(cfg):
    """ops::decoder: attention-norm + o_proj + residual + post-norm + MLP.

    `sum = ones` makes the fused attention-norm a no-op, so this reduces to the
    reference Decoder(x, rx). Exercises the MLP down-projection gather."""
    N, G, D = cfg.num_key_value_heads, 2, cfg.head_dim
    Q = cfg.num_attention_heads * cfg.head_dim

    layer = Decoder(cfg).eval().bfloat16()
    kernel = npu("ops::decoder")

    x = sample(Q)                 # attention output (already "normalized")  m![Q]
    rx = sample(cfg.hidden_size)  # residual stream                         m![H]
    with torch.no_grad():
        expected = layer(x.view(1, 1, Q), rx.view(1, 1, cfg.hidden_size))[0, 0]

    sum_npu = ones(N, G).float().to(DEVICE)          # no-op norm
    x_npu = x.view(N, G, D).to(DEVICE)               # x       m![N, G, D]
    rx_npu = rx.clone().to(DEVICE)                   # rx_hbm  m![H] (in/out)

    kernel(
        x_npu, sum_npu, rx_npu,
        layer.o_proj.weight.to(DEVICE),
        layer.post_attention_layernorm.weight.to(DEVICE),
        layer.mlp.up_proj.weight.to(DEVICE),
        layer.mlp.gate_proj.weight.to(DEVICE),
        layer.mlp.down_proj.weight.to(DEVICE),
    )

    return report("decoder", expected, rx_npu.cpu())


def test_final_layer(cfg):
    """ops::final_layer: final rmsnorm + lm_head projection.

    Only the real vocab logits are checked: the kernel has no masking input to
    guarantee blanked-out values for the `Wp - W` padding rows, so their
    contents aren't part of its contract."""
    Wp = 19 * 8192  # padded vocab size (axes.rs Wp), tiled as 19 chunks of 8192 rows
    W, H = cfg.vocab_size, cfg.hidden_size

    rms = RMSNorm(H, cfg.rms_norm_eps).eval().bfloat16()
    lm_head = nn.Linear(H, W, bias=False).eval().bfloat16()
    kernel = npu("ops::final_layer")

    x = sample(H)
    with torch.no_grad():
        expected = lm_head(rms(x))  # (W,)

    out = sample(Wp).to(DEVICE)
    kernel(
        x.to(DEVICE),
        rms.weight.to(DEVICE),
        lm_head.weight.to(DEVICE),
        out,
    )

    return report("final_layer (vocab)", expected, out.cpu()[:W])


if __name__ == "__main__":
    cfg = Qwen3Config()
    print("Qwen3 NPU kernel tests (random weights; checked against a PyTorch reference)\n")

    results = {}

    print("-- kernels --")
    results.update({
        "projection": test_projection(cfg),
        "attention": test_attention(cfg),
        "decoder": test_decoder(cfg),
        "final_layer": test_final_layer(cfg),
    })

    print()
    for name, ok in results.items():
        print(f"{name:26} {'PASS' if ok else 'FAIL'}")
