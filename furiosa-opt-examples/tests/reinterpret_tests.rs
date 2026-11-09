use furiosa_opt_examples::reinterpret::{R, S, W, reinterpret};
use furiosa_opt_std::prelude::*;

/// FP32 reads of landed bytes return each four bytes' value, whole and from a tile of the
/// reinterpreted view.
#[tokio::test]
async fn reads_landed_bytes_as_fp32() -> eyre::Result<()> {
    let mut device = Device::new(reinterpret.topology())?;
    let words = S::SIZE * R::SIZE * W::SIZE / 4;
    let values: Vec<f32> = (0..words).map(|i| i as f32 * 0.5 - 7.25).collect();
    let bytes: Vec<i8> = values.iter().flat_map(|v| v.to_le_bytes()).map(|b| b as i8).collect();
    let input = HostTensor::<i8, m![S, R, W]>::from_vec(bytes)
        .to_hbm::<m![1], m![S, R, W]>(&mut device.pdma)
        .await?;
    let mut halves = HostTensor::<f32, m![S, R, W / 4]>::from_vec(vec![-1.0; words])
        .to_hbm::<m![1], m![S, R, W / 4]>(&mut device.pdma)
        .await?;

    let whole = launch(reinterpret, (&mut device, &input, &mut halves)).await?;

    let whole = whole.to_host::<m![S, R, W / 4]>(&mut device.pdma).await?.into_vec();
    assert_eq!(whole, values);
    let halves = halves.to_host::<m![S, R, W / 4]>(&mut device.pdma).await?.into_vec();
    let row = W::SIZE / 4;
    for (at, (&got, &value)) in halves.iter().zip(&values).enumerate() {
        let scale = at / row % R::SIZE >= R::SIZE / 2;
        assert_eq!(got, if scale { value } else { -1.0 }, "word {at}");
    }
    Ok(())
}
