#![feature(register_tool, proc_macro_hygiene)]
#![register_tool(furiosa_opt)]

use furiosa_opt_examples::memory_op::{A, B, Q, T, commit_view_bottom_pad, dm_relayout};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::test]
async fn test_dm_relayout() -> eyre::Result<()> {
    let mut device = Device::new(dm_relayout.topology())?;
    let hbm: HbmTensor<i32, m![1], m![A, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..256 * 4096).collect::<Vec<_>>())
            .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
            .await?;
    launch(dm_relayout, (&mut device, &hbm)).await?;

    Ok(())
}

/// `commit_view` into a down-padded tiled `view_mut` — checks bottom padding.
#[tokio::test]
async fn test_commit_view_bottom_pad() -> eyre::Result<()> {
    let mut device = Device::new(commit_view_bottom_pad.topology())?;

    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<f32, m![T, Q % 56 = 8]>::rand(&mut rng)
        .to_hbm::<m![1], m![T, Q % 56 = 8]>(&mut device.pdma)
        .await?;

    launch(commit_view_bottom_pad, (&mut device, &input)).await?;

    Ok(())
}

mod slices {
    use furiosa_opt_std::prelude::*;

    axes![Block = 16, Width = 128, Count = 32];
    type Pool = Slice<BlockTensor>;
    type BlockTensor = HbmTensor<bf16, m![1], m![Block, Width]>;
    type Index = HbmTensor<i32, m![1], m![Count]>;
    type BlockValues = DmTensor<bf16, m![1], m![1 # 2], m![Count, Block / 2], m![Block % 2, Width]>;
    type TokenValues = DmTensor<bf16, m![1], m![1 # 2], m![Count, Width / 16], m![Width % 16]>;
    type Output = HbmTensor<bf16, m![1], m![Count, Block, Width]>;
    axes![Copies = 32768];
    type Copy = HbmTensor<i32, m![1], m![Copies, Count]>;

    #[device(chip = 1)]
    fn gather_produced(
        device: &mut Device,
        blocks: &Pool,
        source: &Copy,
        copied: &mut Copy,
        index: &mut Index,
    ) -> Output {
        source.view().to_hbm_view(&mut device.tdma, copied.view_mut());
        copied
            .view()
            .tile::<m![Copies], 1, m![Copies = 1 # Copies, Count]>(Copies::SIZE - 1)
            .to_hbm_view(&mut device.tdma, index.view_mut());
        let values: BlockValues = blocks
            .gather::<m![1], m![Block, Width]>()
            .by_byte_offsets(index)
            .to_dm(&mut device.tdma);
        values.to_hbm(&mut device.tdma)
    }

    #[tokio::test]
    async fn reads_produced_indices() -> eyre::Result<()> {
        let mut device = Device::new(gather_produced.topology())?;
        let block =
            HostTensor::<bf16, m![Block, Width]>::from_vec(vec![bf16::from_f32(1.0); Block::SIZE * Width::SIZE])
                .to_hbm(&mut device.pdma)
                .await?;
        let blocks = Slice::from(vec![block]);
        let source: Copy = HostTensor::<i32, m![Copies, Count]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
        let mut copied = Copy::alloc(&device)?;
        let mut index: Index = HostTensor::<i32, m![Count]>::from_vec(vec![i32::MAX; Count::SIZE])
            .to_hbm(&mut device.pdma)
            .await?;
        let output = launch(
            gather_produced,
            (&mut device, &blocks, &source, &mut copied, &mut index),
        )
        .await?;
        let values = output
            .to_host::<m![Count, Block, Width]>(&mut device.pdma)
            .await?
            .into_vec();
        assert!(values.into_iter().all(|value| value == bf16::from_f32(1.0)));
        Ok(())
    }

    #[device(chip = 1)]
    fn select(device: &mut Device, blocks: &Pool) -> BlockTensor {
        blocks[0].to_hbm(&mut device.tdma)
    }

    #[device(chip = 1)]
    fn gather(device: &mut Device, blocks: &Pool, index: &Index, length: &HbmTensor<i32, m![1], m![1]>) -> Output {
        let index: Index = index.to_hbm(&mut device.tdma);
        let values: BlockValues = blocks
            .gather::<m![1], m![Block, Width]>()
            .by_byte_offsets(&index)
            .prefix::<m![Count], _>(length, 0)
            .to_dm(&mut device.tdma);
        values.to_hbm(&mut device.tdma)
    }

    #[device(chip = 1)]
    fn scatter(
        device: &mut Device,
        blocks: &mut Pool,
        index: &Index,
        updates: &HbmTensor<bf16, m![1], m![Count, Width]>,
    ) {
        let values: TokenValues = updates.to_dm(&mut device.tdma);
        blocks
            .scatter::<m![Block], m![Width]>()
            .by_byte_offsets(index.view())
            .from_dm(&mut device.tdma, values);
    }

    #[cfg(backend = "npu")]
    #[tokio::test]
    async fn rejects_invalid_indices() -> eyre::Result<()> {
        let mut device = Device::new(gather.topology())?;
        let blocks = Slice::from(vec![BlockTensor::alloc(&device)?]);
        let index = HostTensor::<i32, m![Count]>::from_vec(vec![(Block::SIZE * Width::SIZE * 2) as i32; Count::SIZE])
            .to_hbm(&mut device.pdma)
            .await?;
        let length = HostTensor::<i32, m![1]>::from_vec(vec![Count::SIZE as i32])
            .to_hbm(&mut device.pdma)
            .await?;
        let Err(error) = launch(gather, (&mut device, &blocks, &index, &length)).await else {
            panic!("an out-of-bounds index must fail");
        };
        let message = error.to_string();
        assert!(
            message.contains("Assertion") && message.contains("slice index exceeds block allocation"),
            "{message}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn preserves_changed_membership() -> eyre::Result<()> {
        let mut device = Device::new(gather.topology())?;
        for wide in [false, true] {
            let mut expected: Vec<_> = (1..=4)
                .map(|id| vec![bf16::from_f32(id as f32); Block::SIZE * Width::SIZE])
                .collect();
            let mut blocks = Vec::new();
            let mut gap = None;
            for values in &expected {
                // Exercise addresses wider than a DMA offset on the NPU.
                if cfg!(backend = "npu") && wide && blocks.len() == 1 {
                    gap = Some(HbmTensor::<u8, m![1], m![4294967296]>::alloc(&device)?);
                }
                let mut block = BlockTensor::alloc(&device)?;
                HostTensor::<bf16, m![Block, Width]>::from_vec(values.clone())
                    .to_hbm(&mut device.pdma)
                    .output(&mut block)
                    .await?;
                blocks.push(block);
            }
            let mut blocks = Slice::from(blocks);
            let reads = HostTensor::<i32, m![Count]>::from_vec(
                (0..Count::SIZE).map(|i| ((i % 4) * Block::SIZE * Width::SIZE * 2) as i32),
            )
            .to_hbm(&mut device.pdma)
            .await?;
            let writes = HostTensor::<i32, m![Count]>::from_vec(
                (0..Count::SIZE).map(|i| (((i % 4) * Block::SIZE + i / 4) * Width::SIZE * 2) as i32),
            )
            .to_hbm(&mut device.pdma)
            .await?;

            for (phase, valid) in [1, 17, 32, 32, 0].into_iter().enumerate() {
                if phase == 1 {
                    blocks.swap(0, 3);
                    expected.swap(0, 3);
                } else if phase == 3 {
                    blocks.truncate(3);
                    blocks.extend([BlockTensor::alloc(&device)?]);
                    HostTensor::<bf16, m![Block, Width]>::from_vec(expected[3].clone())
                        .to_hbm(&mut device.pdma)
                        .output(&mut blocks[3])
                        .await?;
                } else if phase == 2 {
                    blocks[1] = BlockTensor::alloc(&device)?;
                    expected[1].fill(bf16::from_f32(9.0));
                    HostTensor::<bf16, m![Block, Width]>::from_vec(expected[1].clone())
                        .to_hbm(&mut device.pdma)
                        .output(&mut blocks[1])
                        .await?;
                }
                let selected = launch(select, (&mut device, &blocks)).await?;
                assert_eq!(
                    selected.to_host::<m![Block, Width]>(&mut device.pdma).await?.into_vec(),
                    expected[0],
                );
                let length = HostTensor::<i32, m![1]>::from_vec(vec![valid as i32])
                    .to_hbm(&mut device.pdma)
                    .await?;
                let result = launch(gather, (&mut device, &blocks, &reads, &length)).await?;
                let result = result
                    .to_host::<m![Count, Block, Width]>(&mut device.pdma)
                    .await?
                    .into_vec();
                let prefix: Vec<_> = (0..valid).flat_map(|i| expected[i % 4].iter().copied()).collect();
                assert_eq!(&result[..prefix.len()], prefix);

                let update = HostTensor::<bf16, m![Count, Width]>::from_vec(
                    (0..Count::SIZE * Width::SIZE)
                        .map(|i| bf16::from_f32((i / Width::SIZE + phase * Count::SIZE + 33) as f32)),
                )
                .to_hbm(&mut device.pdma)
                .await?;
                launch(scatter, (&mut device, &mut blocks, &writes, &update)).await?;
                for i in 0..Count::SIZE {
                    expected[i % 4][(i / 4) * Width::SIZE..(i / 4 + 1) * Width::SIZE]
                        .fill(bf16::from_f32((i + phase * Count::SIZE + 33) as f32));
                }
                for (block, expected) in blocks.iter().zip(&expected) {
                    assert_eq!(
                        block.to_host::<m![Block, Width]>(&mut device.pdma).await?.into_vec(),
                        *expected
                    );
                }
            }
            drop(gap);
        }
        Ok(())
    }
}

mod batched_slices {
    use furiosa_opt_std::prelude::*;

    // Multiple DMA bundles exhaust and reuse the completion-ID window.
    axes![Rows = 256, Count = 4, Block = 16, Width = 128, Windows = 2];
    type KvBlock = HbmTensor<bf16, m![1], m![Block, Width]>;
    type Indices = HbmTensor<i32, m![1], m![Rows, Count]>;
    type Lengths = HbmTensor<i32, m![1], m![Rows, Windows]>;
    type Output = HbmTensor<bf16, m![1], m![Rows, Count, Block, Width]>;

    #[device(chip = 1)]
    fn gather(device: &mut Device, blocks: &Slice<KvBlock>, indices: &Indices, lengths: &Lengths) -> Output {
        let mut selected: HbmTensor<i32, m![1], m![Rows]> = HbmTensor::new();
        lengths
            .view()
            .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows]>(1)
            .to_hbm_view(&mut device.tdma, selected.view_mut());
        let values = blocks
            .gather::<m![1], m![Block, Width]>()
            .by_byte_offsets(indices)
            .prefix::<m![Count], _>(&selected, 0)
            .to_dm::<m![1 # 2], m![Rows / 32, Count, Block / 2], m![Rows % 32, Block % 2, Width]>(&mut device.tdma);
        values.to_hbm(&mut device.tdma)
    }

    #[tokio::test]
    async fn respects_each_rows_prefix() -> eyre::Result<()> {
        let mut device = Device::new(gather.topology())?;
        for wide in [false, true] {
            let mut blocks = Vec::new();
            let mut gap = None;
            for value in [7.0, 11.0] {
                if cfg!(backend = "npu") && wide && !blocks.is_empty() {
                    gap = Some(HbmTensor::<u8, m![1], m![4294967296]>::alloc(&device)?);
                }
                blocks.push(
                    HostTensor::<bf16, m![Block, Width]>::from_vec(vec![
                        bf16::from_f32(value);
                        Block::SIZE * Width::SIZE
                    ])
                    .to_hbm(&mut device.pdma)
                    .await?,
                );
            }
            let blocks = Slice::from(blocks);
            let mut indices = Indices::alloc(&device)?;
            let mut counts = Lengths::alloc(&device)?;
            // Reuse the allocations while changing both the live prefix and selected blocks.
            for iteration in 0..32 {
                let lengths = (0..Rows::SIZE)
                    .map(|row| {
                        let row = if iteration % 2 == 0 { 0 } else { row };
                        ((row + iteration) % (Count::SIZE + 1)) as i32
                    })
                    .collect::<Vec<_>>();
                HostTensor::<i32, m![Rows, Count]>::from_vec(lengths.iter().enumerate().flat_map(|(row, &length)| {
                    (0..Count::SIZE).map(move |at| {
                        if at < length as usize {
                            ((row + iteration) % 2 * Block::SIZE * Width::SIZE * 2) as i32
                        } else {
                            i32::MAX
                        }
                    })
                }))
                .to_hbm(&mut device.pdma)
                .output(&mut indices)
                .await?;
                HostTensor::<i32, m![Rows, Windows]>::from_vec(lengths.iter().flat_map(|&length| [-1, length]))
                    .to_hbm(&mut device.pdma)
                    .output(&mut counts)
                    .await?;
                let values = launch(gather, (&mut device, &blocks, &indices, &counts))
                    .await?
                    .to_host::<m![Rows, Count, Block, Width]>(&mut device.pdma)
                    .await?
                    .into_vec();
                for (row, length) in lengths.into_iter().enumerate() {
                    let start = row * Count::SIZE * Block::SIZE * Width::SIZE;
                    let expected = if (row + iteration) % 2 == 0 { 7.0 } else { 11.0 };
                    assert!(
                        values[start..start + length as usize * Block::SIZE * Width::SIZE]
                            .iter()
                            .all(|v| *v == bf16::from_f32(expected)),
                        "iteration {iteration}, row {row}"
                    );
                }
            }
            drop(gap);
        }
        Ok(())
    }
}
