use furiosa_opt_examples::vrf_operand::{
    B, N, V, W, X, Y, vrf_slice_regroup, vrf_slice_rename, vrf_slice_reshape, vrf_slice_reshape_64,
    way4_packet_broadcast, way4_packet_consecutive, way8_packet_broadcast, way8_packet_broadcast_span,
    way8_packet_consecutive,
};
use furiosa_opt_std::prelude::*;

/// The reshaped operand feeds every slice: each `W` row reads the same `B` vector, which is the
/// broadcast the operand was written as. Rejecting the un-reshaped operand is a compile error, so it
/// lives as a `compile_fail` example in `negative::vector_engine`'s module docs instead.
#[tokio::test]
async fn test_vrf_slice_reshape() -> eyre::Result<()> {
    let mut device = Device::new(vrf_slice_reshape.topology())?;

    let input = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).collect::<Vec<_>>());
    let operand = HostTensor::<i32, m![B]>::from_vec((0..<m![B]>::SIZE as i32).map(|x| x * 10).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![W, B]>(&mut device.pdma).await?;
    let operand_hbm = operand.to_hbm::<m![1], m![B]>(&mut device.pdma).await?;

    let output = launch(vrf_slice_reshape, (&mut device, &input_hbm, &operand_hbm)).await?;

    let expected: Vec<i32> = (0..<m![W, B]>::SIZE as i32)
        .map(|x| x + (x % <m![B]>::SIZE as i32) * 10)
        .collect();
    assert_eq!(output.to_host::<m![W, B]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

/// [`test_vrf_slice_reshape`] at the 64-slice topology, the one a default config has.
#[tokio::test]
async fn test_vrf_slice_reshape_64() -> eyre::Result<()> {
    let mut device = Device::new(vrf_slice_reshape_64.topology())?;

    let input = HostTensor::<i32, m![V, B]>::from_vec((0..<m![V, B]>::SIZE as i32).collect::<Vec<_>>());
    let operand = HostTensor::<i32, m![B]>::from_vec((0..<m![B]>::SIZE as i32).map(|x| x * 10).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, B]>(&mut device.pdma).await?;
    let operand_hbm = operand.to_hbm::<m![1], m![B]>(&mut device.pdma).await?;

    let output = launch(vrf_slice_reshape_64, (&mut device, &input_hbm, &operand_hbm)).await?;

    let expected: Vec<i32> = (0..<m![V, B]>::SIZE as i32)
        .map(|x| x + (x % <m![B]>::SIZE as i32) * 10)
        .collect();
    assert_eq!(output.to_host::<m![V, B]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

/// A composite `Slice` meets the partition rule like a plain one. The regroup decomposes the axis
/// the stream itself names, so slice `w` holds row `w` either way; [`test_vrf_slice_rename`] is where
/// the reshape could move a row and the answer key would notice.
#[tokio::test]
async fn test_vrf_slice_regroup() -> eyre::Result<()> {
    let mut device = Device::new(vrf_slice_regroup.topology())?;

    let cell = |i: i32| (i / <m![B]>::SIZE as i32) * 1000 + i % <m![B]>::SIZE as i32;
    let input = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).collect::<Vec<_>>());
    let operand = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).map(cell).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![W, B]>(&mut device.pdma).await?;
    let operand_hbm = operand.to_hbm::<m![1], m![W, B]>(&mut device.pdma).await?;

    let output = launch(vrf_slice_regroup, (&mut device, &input_hbm, &operand_hbm)).await?;

    let expected: Vec<i32> = (0..<m![W, B]>::SIZE as i32).map(|i| i + cell(i)).collect();
    assert_eq!(output.to_host::<m![W, B]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

/// The reshape renames the 256 slices from `X` / `Y` onto the stream's `W`, so slice `x * 16 + y`
/// must still read the operand row it holds. Every slice holds a different row, so swapping the two
/// axes, or shifting the identity at all, changes the answer instead of hiding behind replication.
#[tokio::test]
async fn test_vrf_slice_rename() -> eyre::Result<()> {
    let mut device = Device::new(vrf_slice_rename.topology())?;

    let cell = |i: i32| (i / <m![B]>::SIZE as i32) * 1000 + i % <m![B]>::SIZE as i32;
    let input = HostTensor::<i32, m![W, B]>::from_vec((0..<m![W, B]>::SIZE as i32).collect::<Vec<_>>());
    // `X` / `Y` row `x * 16 + y` is laid out where `W` row `x * 16 + y` is, which is what the rename
    // claims, so the two tensors index the same way and the answer key is the regroup's.
    let operand =
        HostTensor::<i32, m![X, Y, B]>::from_vec((0..<m![X, Y, B]>::SIZE as i32).map(cell).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![W, B]>(&mut device.pdma).await?;
    let operand_hbm = operand.to_hbm::<m![1], m![X, Y, B]>(&mut device.pdma).await?;

    let output = launch(vrf_slice_rename, (&mut device, &input_hbm, &operand_hbm)).await?;

    let expected: Vec<i32> = (0..<m![W, B]>::SIZE as i32).map(|i| i + cell(i)).collect();
    assert_eq!(output.to_host::<m![W, B]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

#[tokio::test]
async fn test_way4_packet_broadcast() -> eyre::Result<()> {
    let mut device = Device::new(way4_packet_broadcast.topology())?;

    // Powers of two: the division is exact, so the comparison needs no tolerance.
    let scale_of = |group: usize| (1 << (group % 4)) as f32;
    let input = HostTensor::<f32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE).map(|i| i as f32).collect::<Vec<_>>());
    let scale =
        HostTensor::<f32, m![V, N / 16]>::from_vec((0..<m![V, N / 16]>::SIZE).map(scale_of).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, N]>(&mut device.pdma).await?;
    let scale_hbm = scale.to_hbm::<m![1], m![V, N / 16]>(&mut device.pdma).await?;

    let output = launch(way4_packet_broadcast, (&mut device, &input_hbm, &scale_hbm)).await?;

    let expected: Vec<f32> = (0..<m![V, N]>::SIZE).map(|i| i as f32 / scale_of(i / 16)).collect();
    assert_eq!(output.to_host::<m![V, N]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

#[tokio::test]
async fn test_way4_packet_consecutive() -> eyre::Result<()> {
    let mut device = Device::new(way4_packet_consecutive.topology())?;

    // Power-of-two divisors make the comparison exact.
    let scale_of = |i: usize| (1 << (i % 4)) as f32;
    let input = HostTensor::<f32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE).map(|i| i as f32).collect::<Vec<_>>());
    let scale = HostTensor::<f32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE).map(scale_of).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, N]>(&mut device.pdma).await?;
    let scale_hbm = scale.to_hbm::<m![1], m![V, N]>(&mut device.pdma).await?;

    let output = launch(way4_packet_consecutive, (&mut device, &input_hbm, &scale_hbm)).await?;

    let expected: Vec<f32> = (0..<m![V, N]>::SIZE).map(|i| i as f32 / scale_of(i)).collect();
    assert_eq!(output.to_host::<m![V, N]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

#[tokio::test]
async fn test_way8_packet_consecutive() -> eyre::Result<()> {
    let mut device = Device::new(way8_packet_consecutive.topology())?;

    let bias_of = |i: usize| i as i32 * 1000 + 1;
    let input = HostTensor::<i32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE as i32).collect::<Vec<_>>());
    let bias = HostTensor::<i32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE).map(bias_of).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, N]>(&mut device.pdma).await?;
    let bias_hbm = bias.to_hbm::<m![1], m![V, N]>(&mut device.pdma).await?;

    let output = launch(way8_packet_consecutive, (&mut device, &input_hbm, &bias_hbm)).await?;

    let expected: Vec<i32> = (0..<m![V, N]>::SIZE).map(|i| i as i32 + bias_of(i)).collect();
    assert_eq!(output.to_host::<m![V, N]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

#[tokio::test]
async fn test_way8_packet_broadcast() -> eyre::Result<()> {
    let mut device = Device::new(way8_packet_broadcast.topology())?;

    let bias_of = |group: usize| group as i32 * 1000 + 1;
    let input = HostTensor::<i32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE as i32).collect::<Vec<_>>());
    let bias = HostTensor::<i32, m![V, N / 8]>::from_vec((0..<m![V, N / 8]>::SIZE).map(bias_of).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, N]>(&mut device.pdma).await?;
    let bias_hbm = bias.to_hbm::<m![1], m![V, N / 8]>(&mut device.pdma).await?;

    let output = launch(way8_packet_broadcast, (&mut device, &input_hbm, &bias_hbm)).await?;

    let expected: Vec<i32> = (0..<m![V, N]>::SIZE).map(|i| i as i32 + bias_of(i / 8)).collect();
    assert_eq!(output.to_host::<m![V, N]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

#[tokio::test]
async fn test_way8_packet_broadcast_span() -> eyre::Result<()> {
    let mut device = Device::new(way8_packet_broadcast_span.topology())?;

    let bias_of = |group: usize| group as i32 * 1000 + 1;
    let input = HostTensor::<i32, m![V, N]>::from_vec((0..<m![V, N]>::SIZE as i32).collect::<Vec<_>>());
    let bias = HostTensor::<i32, m![V, N / 16]>::from_vec((0..<m![V, N / 16]>::SIZE).map(bias_of).collect::<Vec<_>>());

    let input_hbm = input.to_hbm::<m![1], m![V, N]>(&mut device.pdma).await?;
    let bias_hbm = bias.to_hbm::<m![1], m![V, N / 16]>(&mut device.pdma).await?;

    let output = launch(way8_packet_broadcast_span, (&mut device, &input_hbm, &bias_hbm)).await?;

    let expected: Vec<i32> = (0..<m![V, N]>::SIZE).map(|i| i as i32 + bias_of(i / 16)).collect();
    assert_eq!(output.to_host::<m![V, N]>(&mut device.pdma).await?.into_vec(), expected);

    Ok(())
}

/// Checks multi-axis VRF reads against host-computed values.
mod multi_axis {
    use furiosa_opt_examples::vrf_operand::multi_axis::{
        A, B, C, S, T, way4_axes_broadcast, way4_axes_consecutive, way8_axes_broadcast, way8_axes_consecutive,
    };
    use furiosa_opt_std::prelude::*;

    #[tokio::test]
    async fn test_way8_axes_consecutive() -> eyre::Result<()> {
        let mut device = Device::new(way8_axes_consecutive.topology())?;

        let bias_of = |i: usize| i as i32 % 7 - 3;
        let input =
            HostTensor::<i32, m![S, T, A, B, C]>::from_vec((0..<m![S, T, A, B, C]>::SIZE as i32).collect::<Vec<_>>());
        let bias = HostTensor::<i32, m![S, T, A, B, C]>::from_vec(
            (0..<m![S, T, A, B, C]>::SIZE).map(bias_of).collect::<Vec<_>>(),
        );

        let input_hbm = input.to_hbm::<m![1], m![S, T, A, B, C]>(&mut device.pdma).await?;
        let bias_hbm = bias.to_hbm::<m![1], m![S, T, A, B, C]>(&mut device.pdma).await?;

        let output = launch(way8_axes_consecutive, (&mut device, &input_hbm, &bias_hbm)).await?;

        let expected: Vec<i32> = (0..<m![S, T, A, B, C]>::SIZE).map(|i| i as i32 + bias_of(i)).collect();
        let host = output.to_host::<m![S, T, A, B, C]>(&mut device.pdma).await?;
        assert_eq!(host.into_vec(), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_way8_axes_broadcast() -> eyre::Result<()> {
        let mut device = Device::new(way8_axes_broadcast.topology())?;

        let bias_of = |i: usize| i as i32 % 5 - 2;
        let input =
            HostTensor::<i32, m![S, T, A, B, C]>::from_vec((0..<m![S, T, A, B, C]>::SIZE as i32).collect::<Vec<_>>());
        let bias = HostTensor::<i32, m![S, T]>::from_vec((0..<m![S, T]>::SIZE).map(bias_of).collect::<Vec<_>>());

        let input_hbm = input.to_hbm::<m![1], m![S, T, A, B, C]>(&mut device.pdma).await?;
        let bias_hbm = bias.to_hbm::<m![1], m![S, T]>(&mut device.pdma).await?;

        let output = launch(way8_axes_broadcast, (&mut device, &input_hbm, &bias_hbm)).await?;

        let packet = <m![A, B, C]>::SIZE;
        let expected: Vec<i32> = (0..<m![S, T, A, B, C]>::SIZE)
            .map(|i| i as i32 + bias_of(i / packet))
            .collect();
        let host = output.to_host::<m![S, T, A, B, C]>(&mut device.pdma).await?;
        assert_eq!(host.into_vec(), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_way4_axes_consecutive() -> eyre::Result<()> {
        let mut device = Device::new(way4_axes_consecutive.topology())?;

        let scale_of = |i: usize| (1u32 << (i % 13)) as f32;
        let input = HostTensor::<f32, m![S, T, A, B, C]>::from_vec(
            (0..<m![S, T, A, B, C]>::SIZE).map(|i| i as f32).collect::<Vec<_>>(),
        );
        let scale = HostTensor::<f32, m![S, T, A, B, C]>::from_vec(
            (0..<m![S, T, A, B, C]>::SIZE).map(scale_of).collect::<Vec<_>>(),
        );

        let input_hbm = input.to_hbm::<m![1], m![S, T, A, B, C]>(&mut device.pdma).await?;
        let scale_hbm = scale.to_hbm::<m![1], m![S, T, A, B, C]>(&mut device.pdma).await?;

        let output = launch(way4_axes_consecutive, (&mut device, &input_hbm, &scale_hbm)).await?;

        let expected: Vec<f32> = (0..<m![S, T, A, B, C]>::SIZE).map(|i| i as f32 / scale_of(i)).collect();
        let host = output.to_host::<m![S, T, A, B, C]>(&mut device.pdma).await?;
        assert_eq!(host.into_vec(), expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_way4_axes_broadcast() -> eyre::Result<()> {
        let mut device = Device::new(way4_axes_broadcast.topology())?;

        let scale_of = |i: usize| (1u32 << (i % 13)) as f32;
        let input = HostTensor::<f32, m![S, T, A, B, C]>::from_vec(
            (0..<m![S, T, A, B, C]>::SIZE).map(|i| i as f32).collect::<Vec<_>>(),
        );
        let scale =
            HostTensor::<f32, m![S, T, A]>::from_vec((0..<m![S, T, A]>::SIZE).map(scale_of).collect::<Vec<_>>());

        let input_hbm = input.to_hbm::<m![1], m![S, T, A, B, C]>(&mut device.pdma).await?;
        let scale_hbm = scale.to_hbm::<m![1], m![S, T, A]>(&mut device.pdma).await?;

        let output = launch(way4_axes_broadcast, (&mut device, &input_hbm, &scale_hbm)).await?;

        let packet = <m![B, C]>::SIZE;
        let expected: Vec<f32> = (0..<m![S, T, A, B, C]>::SIZE)
            .map(|i| i as f32 / scale_of(i / packet))
            .collect();
        let host = output.to_host::<m![S, T, A, B, C]>(&mut device.pdma).await?;
        assert_eq!(host.into_vec(), expected);
        Ok(())
    }
}
