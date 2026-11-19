use furiosa_opt_examples::spm_scalar::*;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn scalar_spm_hbm_host_round_trip() -> eyre::Result<()> {
    let mut device = Device::new(initialize.topology())?;
    let scale = HbmScalar::from_host(71i32, &mut device.pdma).await?;
    let output = launch(initialize, (&mut device, &scale)).await?;
    let actual = output.to_host::<m![E]>(&mut device.pdma).await?.into_vec();
    let expected = (0..16).map(|i| 71 + i).collect::<Vec<_>>();
    assert_eq!(actual, expected);
    let output = launch(update, (&mut device, &output, &scale)).await?;
    assert_eq!(
        output.to_host::<m![E]>(&mut device.pdma).await?.into_vec(),
        (0..16).map(|i| 142 + i).collect::<Vec<_>>()
    );
    Ok(())
}

#[tokio::test]
async fn dma_supports_distributed_and_replicated_pe_payloads() -> eyre::Result<()> {
    let mut device = Device::new(dma_distributed.topology())?;
    let expected = (0..64).map(|i| 100 + i * 13).collect::<Vec<_>>();
    let input: Distributed = HostTensor::<i32, m![P, E]>::from_vec(expected.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(dma_distributed, (&mut device, &input)).await?;
    assert_eq!(output.to_host::<m![P, E]>(&mut device.pdma).await?.into_vec(), expected);
    let output = launch(dma_dm, (&mut device, &input)).await?;
    assert_eq!(output.to_host::<m![P, E]>(&mut device.pdma).await?.into_vec(), expected);
    let output = launch(dma_redistribute, (&mut device, &input)).await?;
    assert_eq!(output.to_host::<m![P, E]>(&mut device.pdma).await?.into_vec(), expected);
    let output = launch(distributed_to_scalar, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![P, E]>(&mut device.pdma).await?.into_vec(),
        expected.iter().map(|v| v + 7).collect::<Vec<_>>()
    );
    let scale = HbmScalar::from_host(9i32, &mut device.pdma).await?;
    let output = launch(scalar_to_distributed, (&mut device, &scale)).await?;
    assert_eq!(
        output.to_host::<m![P, E]>(&mut device.pdma).await?.into_vec(),
        (0..64).map(|i| 9 + i).collect::<Vec<_>>()
    );
    let expected = (0..16).map(|i| 200 + i * 17).collect::<Vec<_>>();
    let input: Vector = HostTensor::<i32, m![E]>::from_vec(expected.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(dma_replicated, (&mut device, &input)).await?;
    assert_eq!(output.to_host::<m![E]>(&mut device.pdma).await?.into_vec(), expected);
    let output = launch(replicated_to_scalar, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![E]>(&mut device.pdma).await?.into_vec(),
        expected.iter().map(|v| v + 31).collect::<Vec<_>>()
    );
    let output = launch(scalar_to_replicated, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![E]>(&mut device.pdma).await?.into_vec(),
        expected.iter().map(|v| v + 23).collect::<Vec<_>>()
    );
    Ok(())
}

#[tokio::test]
async fn dma_preserves_distinct_cluster_and_pe_values() -> eyre::Result<()> {
    let mut device = Device::new(dma_clustered.topology())?;
    let expected = (0..128).map(|i| 1000 + i * 37).collect::<Vec<_>>();
    let input: Clustered = HostTensor::<i32, m![C, P, E]>::from_vec(expected.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(dma_clustered, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P, E]>(&mut device.pdma).await?.into_vec(),
        expected
    );
    let output = launch(dma_clustered_redistribute, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P, E]>(&mut device.pdma).await?.into_vec(),
        expected
    );
    Ok(())
}

#[tokio::test]
async fn cluster_local_scalars_and_control_flow() -> eyre::Result<()> {
    let mut device = Device::new(clustered_scalar_update.topology())?;
    let scale = HbmScalar::from_host(17i32, &mut device.pdma).await?;
    let expected = (0..128).map(|i| 1000 + i * 37).collect::<Vec<_>>();
    let input: Clustered = HostTensor::<i32, m![C, P, E]>::from_vec(expected.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(clustered_scalar_update, (&mut device, &input, &scale)).await?;
    assert_eq!(
        output.to_host::<m![C, P, E]>(&mut device.pdma).await?.into_vec(),
        expected.iter().map(|x| x * 3 + 17).collect::<Vec<_>>()
    );
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([1, 20, 30, 40, 3, 50, 60, 70])
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(clustered_varying_index, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(),
        [1, 720, 30, 40, 3, 50, 60, 770]
    );
    Ok(())
}

#[tokio::test]
async fn subtraction_into_new_spm_preserves_source_through_hbm_host() -> eyre::Result<()> {
    let mut device = Device::new(subtract_into_new_spm.topology())?;
    let original = [0, 3, 8, 20, -2, 5, 13, 100];
    let input = HostTensor::<i32, m![C, P]>::from_vec(original)
        .to_hbm::<m![1], m![C, P]>(&mut device.pdma)
        .await?;
    let offset = HbmScalar::from_host(5i32, &mut device.pdma).await?;
    let (source, result) = launch(subtract_into_new_spm, (&mut device, &input, &offset)).await?;
    let source = source.to_host::<m![C, P]>(&mut device.pdma).await?;
    let result = result.to_host::<m![C, P]>(&mut device.pdma).await?;
    assert_eq!(source.into_vec(), original);
    assert_eq!(result.into_vec(), [-5, -2, 3, 15, -7, 0, 8, 95]);
    Ok(())
}

#[tokio::test]
async fn reshape_selects_pe0_without_changing_other_pe_values() -> eyre::Result<()> {
    let mut device = Device::new(reshape_pe0_update.topology())?;
    let original = (0..128).map(|i| i * 7 - 100).collect::<Vec<i32>>();
    let input = HostTensor::<i32, m![C, P, E]>::from_vec(original.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(reshape_pe0_update, (&mut device, &input)).await?;
    let actual = output.to_host::<m![C, P, E]>(&mut device.pdma).await?.into_vec();
    let expected = original
        .into_iter()
        .enumerate()
        .map(|(i, x)| if i % 64 < 16 { x + 1000 } else { x })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn scalar_operators_and_index_widths_match_host() -> eyre::Result<()> {
    let mut device = Device::new(scalar_binary_operators.topology())?;
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([7, 3, 0, 0, 4, 9, 0, 0])
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(scalar_binary_operators, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(),
        [3, 7, 10, 21, 4, 9, 13, 36]
    );
    Ok(())
}

/// Every slot drops by the replicated element the kernel read as a plain value.
#[tokio::test]
async fn uniform_scalar_operand_matches_host() -> eyre::Result<()> {
    let mut device = Device::new(uniform_scalar_operand.topology())?;
    let data: Vec<i32> = (0..16).map(|i| 100 + i * 3).collect();
    let input: Vector = HostTensor::<i32, m![E]>::from_vec(data.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(uniform_scalar_operand, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![E]>(&mut device.pdma).await?.into_vec(),
        data.iter().map(|value| value - data[0]).collect::<Vec<_>>()
    );
    Ok(())
}

/// Each cluster's bias is its own slot 0, so the two clusters shift by different amounts.
#[tokio::test]
async fn a_core_program_reads_its_own_scalar() -> eyre::Result<()> {
    let mut device = Device::new(varying_scalar_operand.topology())?;
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([3, 10, 20, 0, 7, 30, 40, 0])
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(varying_scalar_operand, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(),
        [3, 13, 17, 0, 7, 37, 33, 0]
    );
    Ok(())
}

#[tokio::test]
async fn uniform_scalar_branch_uses_the_host_condition() -> eyre::Result<()> {
    let mut device = Device::new(uniform_scalar_branch.topology())?;
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([1, 20, 30, 40, 3, 50, 60, 70])
        .to_hbm(&mut device.pdma)
        .await?;
    for (taken, expected) in [
        (true, [1, 120, 30, 40, 3, 150, 60, 70]),
        (false, [1, 20, 230, 40, 3, 50, 260, 70]),
    ] {
        let condition = HbmScalar::from_host(taken, &mut device.pdma).await?;
        let output = launch(uniform_scalar_branch, (&mut device, &input, &condition)).await?;
        assert_eq!(output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(), expected);
    }
    Ok(())
}

/// Slot 1 takes the host scalar and slot 2 the constant, both through `SpmScalar::uniform`.
#[tokio::test]
async fn a_uniform_scalar_carries_a_host_value_to_each_cluster() -> eyre::Result<()> {
    let mut device = Device::new(uniform_scalar_seed.topology())?;
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([3, 10, 20, 0, 7, 30, 40, 0])
        .to_hbm(&mut device.pdma)
        .await?;
    let scale = HbmScalar::from_host(5i32, &mut device.pdma).await?;
    let output = launch(uniform_scalar_seed, (&mut device, &input, &scale)).await?;
    assert_eq!(
        output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(),
        [3, 15, 120, 0, 7, 35, 140, 0]
    );
    Ok(())
}

/// Each cluster's sum of slots 1 and 2 lands in slot 3.
#[tokio::test]
async fn a_returned_scalar_is_read_by_the_next_body() -> eyre::Result<()> {
    let mut device = Device::new(scalar_result.topology())?;
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([2, 91, 0, 7, 8, 92, 5, 13])
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(scalar_result, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(),
        [2, 91, 0, 91, 8, 92, 5, 97]
    );
    Ok(())
}

/// Each cluster adds its slots 0 and 1 to the seed of 100.
#[tokio::test]
async fn a_uniform_seed_accumulates_per_cluster() -> eyre::Result<()> {
    let mut device = Device::new(uniform_accumulator.topology())?;
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([2, 91, 0, 7, 8, 92, 5, 13])
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(uniform_accumulator, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(),
        [2, 91, 0, 193, 8, 92, 5, 200]
    );
    Ok(())
}

/// Slot 2 takes twice the cluster's slot 0 plus the shared 7.
#[tokio::test]
async fn a_returned_pair_keeps_each_placement() -> eyre::Result<()> {
    let mut device = Device::new(scalar_result_pair.topology())?;
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec([2, 91, 0, 7, 8, 92, 5, 13])
        .to_hbm(&mut device.pdma)
        .await?;
    let output = launch(scalar_result_pair, (&mut device, &input)).await?;
    assert_eq!(
        output.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(),
        [2, 91, 11, 7, 8, 92, 23, 13]
    );
    Ok(())
}

/// Lengths below `begin` clamp to zero, and lengths past the capacity clamp to it.
#[tokio::test]
async fn gather_window_lengths_clamp_and_replicate() -> eyre::Result<()> {
    let mut device = Device::new(gather_window_lengths.topology())?;
    let original = [0, 3, 8, 20, 5, 6, 13, 100];
    let input: LocalLengths = HostTensor::<i32, m![C, P]>::from_vec(original)
        .to_hbm(&mut device.pdma)
        .await?;
    let begin = HbmScalar::from_host(5i32, &mut device.pdma).await?;
    let capacity = HbmScalar::from_host(8i32, &mut device.pdma).await?;
    let (source, result) = launch(gather_window_lengths, (&mut device, &input, &begin, &capacity)).await?;
    assert_eq!(source.to_host::<m![C, P]>(&mut device.pdma).await?.into_vec(), original);
    assert_eq!(
        result
            .to_host::<m![LengthClusters, LengthPes, C, P]>(&mut device.pdma)
            .await?
            .into_vec(),
        [0, 0, 3, 8, 0, 1, 8, 8].repeat(8)
    );
    Ok(())
}
