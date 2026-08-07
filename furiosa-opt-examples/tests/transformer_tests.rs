//! Run-to-completion smoke tests for the transformer decoder-block `#[device]` entry points,
//! launched on `zero`ed inputs — they only check that a kernel compiles and runs without
//! panicking; numerical correctness lives in `transformer.py verify` (diffs every kernel
//! against the PyTorch reference) — save for the running-max test below, which needs three
//! chunks and so has no single-chunk equivalent in `verify`.

#![allow(clippy::type_complexity)]

use furiosa_opt_examples::transformer::axes::*;
use furiosa_opt_examples::transformer::ops;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_embedding() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(ops::embedding, (&mut *ctx, &input, &mut out)).await;
}

#[tokio::test]
async fn test_projection() {
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_weight = HostTensor::<bf16, m![Q, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_weight = HostTensor::<bf16, m![P, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let v_weight = HostTensor::<bf16, m![P, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let input_rms_weight = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let q_rms_weight = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let k_rms_weight = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let kv_offset = HostTensor::<i32, m![1]>::zero().to_hbm(&mut ctx.pdma).await;
    let cos = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let sin = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut k_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut v_cache = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut q_out = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::projection,
        (
            &mut *ctx,
            &x,
            &q_weight,
            &k_weight,
            &v_weight,
            &input_rms_weight,
            &q_rms_weight,
            &k_rms_weight,
            &kv_offset,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            &mut q_out,
        ),
    )
    .await;
}

#[tokio::test]
async fn test_attention_forward_first() {
    let mut ctx = Context::acquire();

    let q = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let k = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let v = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mask = HostTensor::<f32, m![T]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out_hbm = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::attention_forward_first,
        (&mut *ctx, &q, &k, &v, &mask, &mut max_hbm, &mut sum_hbm, &mut out_hbm),
    )
    .await;
}

#[tokio::test]
async fn test_attention_forward() {
    let mut ctx = Context::acquire();

    let q = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let k = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let v = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mask = HostTensor::<f32, m![T]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out_hbm = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::attention_forward,
        (&mut *ctx, &q, &k, &v, &mask, &mut max_hbm, &mut sum_hbm, &mut out_hbm),
    )
    .await;
}

/// Each `forward` must persist the max it folded, or a later chunk rescales the accumulated sum
/// and output against chunk 0's max. Takes three chunks to see: with two, the stored max and the
/// running one still agree. Only the max is asserted, since that is the state that went missing.
#[tokio::test]
async fn test_attention_forward_persists_running_max() {
    let mut ctx = Context::acquire();

    // Middle chunk's keys of 1 raise the max to `D / sqrt(D)` between two zeroed chunks, so the
    // third chunk sees `sqrt(D)` only if the second one stored it. v/mask stay zero: the max is
    // taken before masking, and nothing else is read back.
    let ones = |n| std::iter::repeat_n(bf16::from_f32(1.0), n);
    let q = HostTensor::<bf16, m![N, G, D]>::from_vec(ones(N::SIZE * G::SIZE * D::SIZE));
    let k_one = HostTensor::<bf16, m![T, N, D]>::from_vec(ones(T::SIZE * N::SIZE * D::SIZE));

    let q = q.to_hbm(&mut ctx.pdma).await;
    let k_one = k_one.to_hbm(&mut ctx.pdma).await;
    let k_zero = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let v = HostTensor::<bf16, m![T, N, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let mask = HostTensor::<f32, m![T]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut max_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut out_hbm = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;

    for (chunk, k) in [&k_zero, &k_one, &k_zero].into_iter().enumerate() {
        let args = (&mut *ctx, &q, k, &v, &mask, &mut max_hbm, &mut sum_hbm, &mut out_hbm);
        match chunk {
            0 => launch(ops::attention_forward_first, args).await,
            _ => launch(ops::attention_forward, args).await,
        }
    }

    let expected = (D::SIZE as f32).sqrt();
    for max in max_hbm.to_host::<m![N, G]>(&mut ctx.pdma).await.into_vec() {
        assert!((max - expected).abs() < 0.05, "running max {max} is not {expected}");
    }
}

#[tokio::test]
async fn test_decoder() {
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![N, G, D]>::zero().to_hbm(&mut ctx.pdma).await;
    let sum_hbm = HostTensor::<f32, m![N, G]>::zero().to_hbm(&mut ctx.pdma).await;
    let mut rx_hbm = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let o_weight = HostTensor::<bf16, m![H, Q]>::zero().to_hbm(&mut ctx.pdma).await;
    let post_rms_weight = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let up_weight = HostTensor::<bf16, m![L, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let gate_weight = HostTensor::<bf16, m![L, H]>::zero().to_hbm(&mut ctx.pdma).await;
    let down_weight = HostTensor::<bf16, m![H, L]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::decoder,
        (
            &mut *ctx,
            &x,
            &sum_hbm,
            &mut rx_hbm,
            &o_weight,
            &post_rms_weight,
            &up_weight,
            &gate_weight,
            &down_weight,
        ),
    )
    .await;
}

#[tokio::test]
async fn test_final_layer() {
    let mut ctx = Context::acquire();

    let input = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let rms_weight = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut ctx.pdma).await;
    let lm_head_weight = HostTensor::<bf16, m![W # 155648 / 8192, W # 155648 % 8192, H]>::zero()
        .to_hbm(&mut ctx.pdma)
        .await;
    let mut out = HostTensor::<bf16, m![Wp]>::zero().to_hbm(&mut ctx.pdma).await;

    launch(
        ops::final_layer,
        (&mut *ctx, &input, &rms_weight, &lm_head_weight, &mut out),
    )
    .await;
}
