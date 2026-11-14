use furiosa_opt_examples::sub_context_engines::{KV, R, S, sub_cast};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Checks vector-engine `i32` to `i8` casting against host truncation.
#[tokio::test]
async fn test_sub_cast() -> eyre::Result<()> {
    let mut device = Device::new(sub_cast.topology())?;

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![S, R, KV]>::rand(&mut rng);
    let input_hbm = input.to_hbm(&mut device.pdma).await?;

    let out_hbm = launch(sub_cast, (&mut device, &input_hbm)).await?;
    let result = out_hbm.to_host::<m![S, R, KV]>(&mut device.pdma).await?;

    let expected: Vec<i8> = input.into_inner().into_vec().into_iter().map(|v| v as i8).collect();
    assert_eq!(expected, result.into_vec());

    Ok(())
}
