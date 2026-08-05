//! Run-to-completion smoke tests for the transformer decoder-block `#[device]` entry points,
//! launched on `zero`ed inputs — they only check that a kernel compiles and runs without
//! panicking; numerical correctness lives in `transformer.py verify` (diffs every kernel
//! against the PyTorch reference).

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
