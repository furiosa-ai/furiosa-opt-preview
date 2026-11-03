use furiosa_opt_examples::view::simpl::view_simpl;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_view_simpl() -> eyre::Result<()> {
    use furiosa_opt_examples::view::simpl::{A, B};

    let mut device = Device::new(view_simpl.topology())?;

    // Create input tensor with shape (A=512)(B=8).
    let input = HostTensor::<i32, m![A, B]>::from_vec((0..4096).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await?;

    // Call the device function.
    let output = launch(view_simpl, (&mut device, &input)).await?;

    // Verify the output tensor content: [[6,7,0,1,2,3,4,5],[14,15,8,9,10,11,12,13],...].
    assert_eq!(
        output.to_host::<m![A, B]>(&mut device.pdma).await?.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(
            (0..512)
                .flat_map(|x| [6, 7, 0, 1, 2, 3, 4, 5].into_iter().map(move |i| 8 * x + i))
                .collect::<Vec<_>>(),
        )
    );

    Ok(())
}

#[tokio::test]
async fn test_write_vrf_view_mut() -> eyre::Result<()> {
    use furiosa_opt_examples::view::whole_view_mut::{A, E, T, write_vrf_view_mut};

    type Layout = m![A, T, E];

    let values: Vec<i32> = (1..=Layout::SIZE as i32).collect();
    let expected: Vec<i32> = values.iter().map(|value| 2 * value).collect();
    let mut device = Device::new(write_vrf_view_mut.topology())?;
    let input = HostTensor::<i32, Layout>::from_vec(values)
        .to_hbm::<m![1], Layout>(&mut device.pdma)
        .await?;

    let output = launch(write_vrf_view_mut, (&mut device, &input)).await?;
    assert_eq!(output.to_host::<Layout>(&mut device.pdma).await?.into_vec(), expected);
    Ok(())
}
