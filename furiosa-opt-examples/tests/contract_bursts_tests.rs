//! Repeated launches of [`contract_bursts`] on the device. Profiling slows the interrupts that
//! retire the bursts' clock-control requests, so run it profiled as well as unprofiled:
//!
//!   FURIOSA_OPT_PROFILE=info \
//!     cargo furiosa-opt test -p furiosa-opt-examples --test contract_bursts_tests
#![cfg(backend = "npu")]

use furiosa_opt_examples::contract_bursts::{A, K, P, R, contract_bursts};
use furiosa_opt_std::prelude::*;
use tracing_subscriber::prelude::*;

/// Every launch of bursts of throttled Tensor Unit operations completes, each output holding the
/// contraction of ones.
#[tokio::test]
async fn completes_every_launch() -> eyre::Result<()> {
    tracing_subscriber::registry().init();
    let mut device = Device::new(contract_bursts.topology())?;
    let input = HostTensor::<i8, m![P, A, K]>::from_vec(vec![1; <m![P, A, K]>::SIZE]);
    let input = input.to_hbm(&mut device.pdma).await?;
    let weight = HostTensor::<i8, m![P, R, K]>::from_vec(vec![1; <m![P, R, K]>::SIZE]);
    let weight = weight.to_hbm(&mut device.pdma).await?;
    for _ in 0..200 {
        let (first, second) = launch(contract_bursts, (&mut device, &input, &weight)).await?;
        for output in [first, second] {
            let values = output.to_host::<m![P, A, R]>(&mut device.pdma).await?.into_vec();
            assert!(values.iter().all(|&value| value == K::SIZE as i32));
        }
    }
    Ok(())
}
