use furiosa_opt_examples::root_view::{Entries, Key, Layers, Width, scatter_layer_rows};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn scatter_preserves_other_layer_entries() -> eyre::Result<()> {
    type Block = HbmTensor<bf16, m![1], m![Width]>;
    let mut device = Device::new(scatter_layer_rows.topology())?;
    let source: Vec<_> = (0..<m![Layers, Key, Width]>::SIZE)
        .map(|value| bf16::from_f32((value % 509) as f32))
        .collect();
    let values = HostTensor::<bf16, m![Layers, Key, Width]>::from_vec(source.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let bytes = i32::try_from(Width::SIZE * std::mem::size_of::<bf16>())?;
    let indices = HostTensor::<i32, m![Key]>::from_vec([2 * bytes, 0])
        .to_hbm(&mut device.pdma)
        .await?;
    let count = HostTensor::<i32, m![1]>::from_vec([i32::try_from(Layers::SIZE)?])
        .to_hbm(&mut device.pdma)
        .await?;
    let mut blocks = (0..Layers::SIZE * Entries::SIZE)
        .map(|_| Block::alloc(&device))
        .collect::<Result<Slice<Block>, _>>()?;
    for block in blocks.iter_mut() {
        HostTensor::<bf16, m![Width]>::from_vec([bf16::from_f32(-1.0); Width::SIZE])
            .to_hbm(&mut device.pdma)
            .output(block)
            .await?;
    }

    launch(
        scatter_layer_rows,
        (&mut device, &values, &indices, &count, &mut blocks),
    )
    .await?;

    for (index, block) in blocks.iter().enumerate() {
        let layer = index / Entries::SIZE;
        let expected = match index % Entries::SIZE {
            0 => source[(layer * Key::SIZE + 1) * Width::SIZE..(layer * Key::SIZE + 2) * Width::SIZE].to_vec(),
            2 => source[layer * Key::SIZE * Width::SIZE..(layer * Key::SIZE + 1) * Width::SIZE].to_vec(),
            _ => vec![bf16::from_f32(-1.0); Width::SIZE],
        };
        assert_eq!(block.to_host::<m![Width]>(&mut device.pdma).await?.into_vec(), expected);
    }
    Ok(())
}
