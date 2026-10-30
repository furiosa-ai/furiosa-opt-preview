//! Direct calls and `FnOnce` adapters must preserve closure captures and arguments.

use furiosa_opt_examples::closure::{Cols, Rows, lend_device, lend_stored, pass_device};
use furiosa_opt_std::prelude::*;

/// Captured inputs must reach their corresponding output arguments in a multi-argument closure call.
#[tokio::test]
async fn copies_through_device_arguments() -> eyre::Result<()> {
    let mut device = Device::new(pass_device.topology())?;
    let first_values: Vec<i32> = (0..Rows::SIZE * Cols::SIZE).map(|i| i as i32 + 1).collect();
    let second_values: Vec<i32> = first_values.iter().map(|value| -value).collect();
    let first_input = HostTensor::<i32, m![Rows, Cols]>::from_vec(first_values.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let second_input = HostTensor::<i32, m![Rows, Cols]>::from_vec(second_values.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut first_output = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut second_output = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    launch(
        pass_device,
        (
            &mut device,
            &first_input,
            &second_input,
            &mut first_output,
            &mut second_output,
        ),
    )
    .await?;
    let actual = first_output.to_host::<m![Rows, Cols]>(&mut device.pdma).await?;
    assert_eq!(actual.into_vec(), first_values);
    let actual = second_output.to_host::<m![Rows, Cols]>(&mut device.pdma).await?;
    assert_eq!(actual.into_vec(), second_values);
    Ok(())
}

/// The inline closure uses the `FnOnce` bound directly, without a closure adapter.
#[tokio::test]
async fn calls_inline_closure_as_fn_once() -> eyre::Result<()> {
    let mut device = Device::new(lend_device.topology())?;
    let expected: Vec<i32> = (0..Rows::SIZE * Cols::SIZE).map(|i| i as i32 + 1).collect();
    let input = HostTensor::<i32, m![Rows, Cols]>::from_vec(expected.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut output = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    launch(lend_device, (&mut device, &input, &mut output)).await?;
    let actual = output.to_host::<m![Rows, Cols]>(&mut device.pdma).await?;
    assert_eq!(actual.into_vec(), expected);
    Ok(())
}

/// The stored `FnMut` closure must preserve its captures when called through a `FnOnce` adapter.
#[tokio::test]
async fn calls_stored_closure_as_fn_once() -> eyre::Result<()> {
    let mut device = Device::new(lend_stored.topology())?;
    let expected: Vec<i32> = (0..Rows::SIZE * Cols::SIZE).map(|i| -(i as i32 + 1)).collect();
    let input = HostTensor::<i32, m![Rows, Cols]>::from_vec(expected.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut output = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    launch(lend_stored, (&mut device, &input, &mut output)).await?;
    let actual = output.to_host::<m![Rows, Cols]>(&mut device.pdma).await?;
    assert_eq!(actual.into_vec(), expected);
    Ok(())
}
