use furiosa_opt_examples::layer_pool::{Block, Entries, Key, Layers, Width, forward, unrolled};
use furiosa_opt_std::prelude::*;

/// Rolled layer accesses match constant-entry accesses and preserve every untouched block.
#[tokio::test]
async fn preserves_layer_pools() -> eyre::Result<()> {
    let mut device = Device::new(forward.topology())?;
    let source = (0..<m![Layers, Key, Width]>::SIZE)
        .map(|index| Ok(bf16::from_f32(f32::from(u16::try_from(index % 251)?))))
        .collect::<eyre::Result<Vec<_>>>()?;
    let values = HostTensor::<bf16, m![Layers, Key, Width]>::from_vec(source.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let bytes = i32::try_from(Width::SIZE * std::mem::size_of::<bf16>())?;
    let indices = HostTensor::<i32, m![Key]>::from_vec([2 * bytes, 0])
        .to_hbm(&mut device.pdma)
        .await?;
    let lengths = HostTensor::<i32, m![1]>::from_vec([i32::try_from(Key::SIZE)?])
        .to_hbm(&mut device.pdma)
        .await?;
    let sentinel = bf16::from_f32(-1.0);
    let mut pools = (0..Layers::SIZE * Entries::SIZE)
        .map(|_| Block::alloc(&device))
        .collect::<Result<Slice<Block>, _>>()?;
    let mut control = (0..Layers::SIZE * Entries::SIZE)
        .map(|_| Block::alloc(&device))
        .collect::<Result<Slice<Block>, _>>()?;

    for live in [0, 1, 2, 3, 4] {
        let count = HostTensor::<i32, m![1]>::from_vec([live])
            .to_hbm(&mut device.pdma)
            .await?;
        for pool in [&mut pools, &mut control] {
            for block in pool.iter_mut() {
                HostTensor::<bf16, m![Width]>::from_vec([sentinel; Width::SIZE])
                    .to_hbm(&mut device.pdma)
                    .output(block)
                    .await?;
            }
        }
        let mut output =
            HostTensor::<bf16, m![Layers, Key, Width]>::from_vec(vec![sentinel; <m![Layers, Key, Width]>::SIZE])
                .to_hbm(&mut device.pdma)
                .await?;
        let mut expected =
            HostTensor::<bf16, m![Layers, Key, Width]>::from_vec(vec![sentinel; <m![Layers, Key, Width]>::SIZE])
                .to_hbm(&mut device.pdma)
                .await?;

        launch(
            forward,
            (
                &mut device,
                &mut pools,
                &values,
                &indices,
                &lengths,
                &count,
                &mut output,
            ),
        )
        .await?;
        launch(
            unrolled,
            (
                &mut device,
                &mut control,
                &values,
                &indices,
                &lengths,
                &count,
                &mut expected,
            ),
        )
        .await?;

        let output = output
            .to_host::<m![Layers, Key, Width]>(&mut device.pdma)
            .await?
            .into_vec();
        assert_eq!(
            output,
            expected
                .to_host::<m![Layers, Key, Width]>(&mut device.pdma)
                .await?
                .into_vec()
        );
        let written = usize::try_from(live)? * Key::SIZE * Width::SIZE;
        assert_eq!(&output[..written], &source[..written]);
        assert!(
            output[written..]
                .iter()
                .all(|value| value.to_bits() == sentinel.to_bits())
        );
        for (index, (block, reference)) in pools.iter().zip(control.iter()).enumerate() {
            let actual = block.to_host::<m![Width]>(&mut device.pdma).await?.into_vec();
            assert_eq!(
                actual,
                reference.to_host::<m![Width]>(&mut device.pdma).await?.into_vec()
            );
            let layer = index / Entries::SIZE;
            let row = match (layer < usize::try_from(live)?, index % Entries::SIZE) {
                (true, 0) => Some(1),
                (true, 2) => Some(0),
                _ => None,
            };
            match row {
                Some(row) => {
                    let start = (layer * Key::SIZE + row) * Width::SIZE;
                    assert_eq!(&actual, &source[start..start + Width::SIZE]);
                }
                None => assert!(actual.iter().all(|value| value.to_bits() == sentinel.to_bits())),
            }
        }
        println!("layers={live}: output and all pool blocks match exactly");
    }
    Ok(())
}
