use crate::common::assert_f32_bits_eq;
use furiosa_opt_examples::vector_engine::{A, ve_cumsum_spans_f32, ve_cumsum_spans_i32};
use furiosa_opt_std::prelude::*;

/// A flit's place in its span of 16 flits, for every element of a row of 64 flits.
fn positions() -> Vec<i32> {
    (0..A::SIZE).map(|element| ((element / 8) % 16) as i32).collect()
}

/// Each lane's running sum over spans of 16 flits, with `add` in stream order.
fn spans<D: Copy>(input: &[D], add: impl Fn(D, D) -> D) -> Vec<D> {
    let mut out = input.to_vec();
    for element in 0..A::SIZE {
        if (element / 8) % 16 != 0 {
            out[element] = add(out[element], out[element - 8]);
        }
    }
    out
}

#[tokio::test]
async fn test_ve_cumsum_spans_i32() -> eyre::Result<()> {
    let mut device = Device::new(ve_cumsum_spans_i32.topology())?;
    let values: Vec<i32> = (0..A::SIZE as i32).map(|e| (e * 7919) % 1000 - 500).collect();
    let position = HostTensor::<i32, m![A]>::from_vec(positions())
        .to_hbm(&mut device.pdma)
        .await?;
    let input = HostTensor::<i32, m![A]>::from_vec(values.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(ve_cumsum_spans_i32, (&mut device, &position, &input)).await?;
    let result = output.to_host::<m![A]>(&mut device.pdma).await?;
    assert_eq!(spans(&values, |x, y| x.saturating_add(y)), result.into_vec());
    Ok(())
}

#[tokio::test]
async fn test_ve_cumsum_spans_f32() -> eyre::Result<()> {
    let mut device = Device::new(ve_cumsum_spans_f32.topology())?;
    let values: Vec<f32> = (0..A::SIZE).map(|e| ((e * 7919) % 1000) as f32 / 37.0 - 13.0).collect();
    let position = HostTensor::<i32, m![A]>::from_vec(positions())
        .to_hbm(&mut device.pdma)
        .await?;
    let input = HostTensor::<f32, m![A]>::from_vec(values.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(ve_cumsum_spans_f32, (&mut device, &position, &input)).await?;
    let result = output.to_host::<m![A]>(&mut device.pdma).await?;
    assert_f32_bits_eq(&spans(&values, |x, y| x + y), &result.into_vec());
    Ok(())
}
