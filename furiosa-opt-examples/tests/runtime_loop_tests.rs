use furiosa_opt_examples::runtime_loop::{Slice, Width, loop_reassign_ref_keeps_original, loop_reassign_then_write};
use furiosa_opt_std::prelude::*;

fn input_host() -> HostTensor<i32, m![Slice, Width]> {
    HostTensor::from_vec((0..<m![Slice, Width]>::SIZE).map(|i| i as i32 % 23 - 11))
}

/// A write to the reassigned tensor adds three per iteration: `output == input + 9`.
#[tokio::test]
async fn test_loop_reassign_then_write() -> eyre::Result<()> {
    let mut device = Device::new(loop_reassign_then_write.topology())?;
    let input = input_host();
    let input_hbm = input.to_hbm(&mut device.pdma).await?;

    let output = launch(loop_reassign_then_write, (&mut device, &input_hbm)).await?;

    assert_eq!(
        input.into_inner().map(|x| x + 9).into_vec(),
        output.to_host::<m![Slice, Width]>(&mut device.pdma).await?.into_vec()
    );
    Ok(())
}

/// The reference ends at the third update while `original` keeps the input: `(input, input + 3)`.
#[tokio::test]
async fn test_loop_reassign_ref_keeps_original() -> eyre::Result<()> {
    let mut device = Device::new(loop_reassign_ref_keeps_original.topology())?;
    let input = input_host();
    let input_hbm = input.to_hbm(&mut device.pdma).await?;

    let (original, carried) = launch(loop_reassign_ref_keeps_original, (&mut device, &input_hbm)).await?;

    let expected = input.into_inner();
    assert_eq!(
        expected.map(|x| x + 3).into_vec(),
        carried.to_host::<m![Slice, Width]>(&mut device.pdma).await?.into_vec()
    );
    assert_eq!(
        expected.into_vec(),
        original.to_host::<m![Slice, Width]>(&mut device.pdma).await?.into_vec()
    );
    Ok(())
}
