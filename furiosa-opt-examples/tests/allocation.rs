#![feature(register_tool, proc_macro_hygiene)]
#![register_tool(furiosa_opt)]

mod allocation {
    use furiosa_opt_std::prelude::*;

    axes![Rows = 4, Width = 256, Picked = 8];
    type Block = HbmTensor<i32, m![1], m![Rows, Width]>;
    type Selected = HbmTensor<i32, m![1], m![Picked, Width % 128]>;

    #[device(chip = 1)]
    fn write(device: &mut Device, source: &Block, blocks: &mut Slice<Block>, index: &HbmScalar<usize>) {
        let index = index.to_spm(&mut device.tdma);
        source.view().to_hbm_view(&mut device.tdma, blocks[index].view_mut());
        source
            .view()
            .tile::<m![Rows], 1, m![Rows = 1 # 4, Width]>(0)
            .to_hbm_view(
                &mut device.tdma,
                blocks[index]
                    .view_mut()
                    .tile::<m![Rows], 1, m![Rows = 1 #{!} 4, Width]>(3),
            );
    }

    #[device(chip = 1)]
    fn read(device: &mut Device, blocks: &Slice<Block>, index: &HbmScalar<usize>) -> Block {
        let index = index.to_spm(&mut device.tdma);
        blocks[index].to_hbm(&mut device.tdma)
    }

    #[tokio::test]
    async fn preserves_allocated_storage() -> eyre::Result<()> {
        let mut device = Device::new(write.topology())?;
        let mut blocks = (0..3).map(|_| Block::alloc(&device)).collect::<Result<Slice<_>, _>>()?;
        let mut output = Block::alloc(&device)?;
        let mut expected = vec![Vec::new(); blocks.len()];
        for phase in 0..2 {
            if phase == 1 {
                blocks.swap(0, 2);
                expected.swap(0, 2);
                blocks[1] = Block::alloc(&device)?;
            }
            for at in [2, 0, 1] {
                let values: Vec<_> = (0..Rows::SIZE * Width::SIZE)
                    .map(|cell| (phase * 10_000 + at * 2_000 + cell) as i32)
                    .collect();
                let source = HostTensor::<i32, m![Rows, Width]>::from_vec(values.clone())
                    .to_hbm(&mut device.pdma)
                    .await?;
                let index = HbmScalar::from_host(at, &mut device.pdma).await?;
                launch(write, (&mut device, &source, &mut blocks, &index)).await?;
                expected[at] = values;
                expected[at].copy_within(0..Width::SIZE, (Rows::SIZE - 1) * Width::SIZE);
            }
            for (at, expected) in expected.iter().enumerate() {
                let index = HbmScalar::from_host(at, &mut device.pdma).await?;
                launch(read, (&mut device, &blocks, &index)).output(&mut output).await?;
                assert_eq!(
                    output.to_host::<m![Rows, Width]>(&mut device.pdma).await?.into_vec(),
                    *expected,
                );
                assert_eq!(
                    blocks[at]
                        .to_host::<m![Rows, Width]>(&mut device.pdma)
                        .await?
                        .into_vec(),
                    *expected,
                );
            }
        }
        Ok(())
    }

    #[device(chip = 1)]
    fn gather(device: &mut Device, table: &Block, offsets: &HbmTensor<i32, m![1], m![Picked]>) -> Selected {
        table
            .view()
            .tile::<m![Width / 128], 1, m![Rows, Width / 128 = 1 # 2, Width % 128]>(1)
            .gather::<m![Rows], m![Width / 128 = 1 # 2, Width % 128]>()
            .by_byte_offsets(offsets)
            .to_dm::<m![1 # 2], m![Picked, Width % 128 / 4], m![Width % 4]>(&mut device.tdma)
            .to_hbm(&mut device.tdma)
    }

    #[tokio::test]
    async fn gathers_tiled_payloads() -> eyre::Result<()> {
        let mut device = Device::new(gather.topology())?;
        let table = HostTensor::<i32, m![Rows, Width]>::from_vec(0..1024)
            .to_hbm(&mut device.pdma)
            .await?;
        let selected = [3, 1, 2, 0, 0, 2, 1, 3];
        let offsets = HostTensor::<i32, m![Picked]>::from_vec(selected.map(|row| row * 1024))
            .to_hbm(&mut device.pdma)
            .await?;
        let output = launch(gather, (&mut device, &table, &offsets))
            .await?
            .to_host::<m![Picked, Width % 128]>(&mut device.pdma)
            .await?
            .into_vec();
        let expected: Vec<_> = selected
            .into_iter()
            .flat_map(|row| row * 256 + 128..(row + 1) * 256)
            .collect();
        assert_eq!(output, expected);
        Ok(())
    }
}
