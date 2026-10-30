//! Dropping a running launch's Future cancels nothing: the launch runs to completion, and holds its
//! memory, past the caller's last handle, and its answer slot until the device answers. Gated to
//! the `npu` backend: the host backend ends a launch before its submission returns.
#![cfg(backend = "npu")]

use std::pin::pin;
use std::task::{Context, Waker};

use furiosa_opt_examples::transformer::axes::{H, W, Wp};
use furiosa_opt_examples::transformer::ops;
use furiosa_opt_std::prelude::*;

type Weight = m![W # 155648 / 8192, W # 155648 % 8192, H];

/// More launches than one cluster's answer slots, so a leaked slot would stall the tail.
const DROPPED: usize = 24;

/// The final layer streams hundreds of megabytes of weight, so every Future drops while its launch
/// runs.
#[tokio::test]
async fn dropped_launches_run_to_completion() -> eyre::Result<()> {
    let mut device = Device::new(ops::final_layer.topology())?;
    let input = HostTensor::<bf16, m![H]>::from_vec((0..H::SIZE).map(|i| bf16::from_f32((i % 7 + 1) as f32 * 0.1)))
        .to_hbm(&mut device.pdma)
        .await?;
    let rms = HostTensor::<bf16, m![H]>::from_vec(vec![bf16::from_f32(1.0); H::SIZE])
        .to_hbm(&mut device.pdma)
        .await?;
    let mut lm_head = vec![bf16::from_f32(0.0); Wp::SIZE * H::SIZE];
    for row in 0..W::SIZE {
        lm_head[row * H::SIZE] = bf16::from_f32((row % 17) as f32 * 0.01 - 0.08);
    }
    let weight = HostTensor::<bf16, Weight>::from_vec(lm_head)
        .to_hbm(&mut device.pdma)
        .await?;
    let mut out = HostTensor::<bf16, m![Wp]>::zero().to_hbm(&mut device.pdma).await?;
    launch(ops::final_layer, (&mut device, &input, &rms, &weight, &mut out)).await?;
    let expected = out.to_host::<m![Wp]>(&mut device.pdma).await?.into_vec();

    macro_rules! drop_running {
        () => {
            for _ in 0..DROPPED {
                let mut submitted =
                    pin!(launch(ops::final_layer, (&mut device, &input, &rms, &weight, &mut out)).submit());
                let polled = submitted.as_mut().poll(&mut Context::from_waker(Waker::noop()));
                assert!(
                    polled.is_pending(),
                    "a launch ended before its Future dropped: {polled:?}"
                );
            }
        };
    }

    drop_running!();
    // The dropped launches still write `out`, and a transfer into it waits for them to end.
    let polled = pin!(out.ready()).poll(&mut Context::from_waker(Waker::noop()));
    assert!(
        polled.is_pending(),
        "no dropped launch holds the output any more: {polled:?}"
    );
    let pattern = vec![bf16::from_f32(-1.0); Wp::SIZE];
    HostTensor::<bf16, m![Wp]>::from_vec(pattern.clone())
        .to_hbm(&mut device.pdma)
        .output(&mut out)
        .await?;
    assert_eq!(out.to_host::<m![Wp]>(&mut device.pdma).await?.into_vec(), pattern);

    launch(ops::final_layer, (&mut device, &input, &rms, &weight, &mut out)).await?;
    assert_eq!(out.to_host::<m![Wp]>(&mut device.pdma).await?.into_vec(), expected);

    // The dropped launches hold `out` past its last handle, so memory allocated meanwhile is
    // never where they still write.
    drop_running!();
    drop(out);
    let other = HostTensor::<bf16, m![Wp]>::from_vec(pattern.clone())
        .to_hbm::<m![1], m![Wp]>(&mut device.pdma)
        .await?;
    device.synchronize().await?;
    assert_eq!(
        other.to_host::<m![Wp]>(&mut device.pdma).await?.into_vec(),
        pattern,
        "a running launch wrote memory allocated after its output's last handle dropped"
    );

    Ok(())
}
