use furiosa_opt_examples::algebraic::{Cols, Rows, columns, result, rows};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn selected_layout_preserves_values() -> eyre::Result<()> {
    let mut device = Device::new(rows.topology())?;
    let values: Vec<i32> = (0..Rows::SIZE * Cols::SIZE).map(|i| i as i32 - 123).collect();
    let input = HostTensor::<i32, m![Rows, Cols]>::from_vec(values.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let row_major = launch(rows, (&mut device, &input)).await?;
    let column_major = launch(columns, (&mut device, &input)).await?;
    let mut fallible = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    launch(result, (&mut device, &input, &mut fallible)).await?;
    for output in [row_major, column_major, fallible] {
        assert_eq!(
            output.to_host::<m![Rows, Cols]>(&mut device.pdma).await?.into_vec(),
            values
        );
    }
    Ok(())
}

#[tokio::test]
async fn array_selection_preserves_distinct_state() -> eyre::Result<()> {
    use furiosa_opt_examples::algebraic::{Block, Entry, Key, update_layers};
    let mut device = Device::new(update_layers.topology())?;
    let mut layers = [
        (0..3)
            .map(|_| Block::alloc(&device))
            .collect::<Result<Slice<Block>, _>>()?,
        (0..3)
            .map(|_| Block::alloc(&device))
            .collect::<Result<Slice<Block>, _>>()?,
    ];
    for (index, layer) in layers.iter_mut().enumerate() {
        for block in layer.iter_mut() {
            HostTensor::<i32, m![Entry]>::from_vec([index as i32; 4])
                .to_hbm(&mut device.pdma)
                .output(block)
                .await?;
        }
    }
    let indices = HostTensor::<i32, m![Key]>::from_vec([0, 32])
        .to_hbm::<m![1], m![Key]>(&mut device.pdma)
        .await?;
    let values = HostTensor::<i32, m![Key, Entry]>::from_vec(100..108)
        .to_hbm::<m![1], m![Key, Entry]>(&mut device.pdma)
        .await?;
    launch(update_layers, (&mut device, &mut layers, &indices, &values)).await?;
    for (index, layer) in layers.iter().enumerate() {
        let mut actual = Vec::new();
        for block in layer.iter() {
            actual.extend(block.to_host::<m![Entry]>(&mut device.pdma).await?.into_vec());
        }
        assert_eq!(
            actual,
            [
                100,
                101,
                102,
                103,
                index as i32,
                index as i32,
                index as i32,
                index as i32,
                104,
                105,
                106,
                107
            ]
        );
    }
    Ok(())
}

#[tokio::test]
async fn slice_array_preserves_distinct_state() -> eyre::Result<()> {
    use furiosa_opt_examples::algebraic::{Block, Entry, Key, update_blocks};
    let mut device = Device::new(update_blocks.topology())?;
    let mut layers = [
        (0..3)
            .map(|_| Block::alloc(&device))
            .collect::<Result<Slice<Block>, _>>()?,
        (0..3)
            .map(|_| Block::alloc(&device))
            .collect::<Result<Slice<Block>, _>>()?,
    ];
    for (index, layer) in layers.iter_mut().enumerate() {
        for block in layer.iter_mut() {
            HostTensor::<i32, m![Entry]>::from_vec([index as i32; 4])
                .to_hbm(&mut device.pdma)
                .output(block)
                .await?;
        }
    }
    let indices = HostTensor::<i32, m![Key]>::from_vec([0, 32])
        .to_hbm::<m![1], m![Key]>(&mut device.pdma)
        .await?;
    let values = HostTensor::<i32, m![Key]>::from_vec(100..102)
        .to_hbm::<m![1], m![Key]>(&mut device.pdma)
        .await?;
    launch(update_blocks, (&mut device, &mut layers, &indices, &values)).await?;
    for (index, layer) in layers.iter().enumerate() {
        let mut actual = Vec::new();
        for block in layer.iter() {
            actual.extend(block.to_host::<m![Entry]>(&mut device.pdma).await?.into_vec());
        }
        assert_eq!(
            actual,
            [
                100,
                index as i32,
                index as i32,
                index as i32,
                index as i32,
                index as i32,
                index as i32,
                index as i32,
                101,
                index as i32,
                index as i32,
                index as i32
            ]
        );
    }
    Ok(())
}

#[tokio::test]
async fn maps_captured_tensors() -> eyre::Result<()> {
    use furiosa_opt_examples::algebraic::map_tensors;
    let mut device = Device::new(map_tensors.topology())?;
    let values: Vec<i32> = (0..Rows::SIZE * Cols::SIZE).map(|i| i as i32 - 123).collect();
    let other: Vec<i32> = values.iter().rev().copied().collect();
    let first = HostTensor::<i32, m![Rows, Cols]>::from_vec(values.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let second = HostTensor::<i32, m![Rows, Cols]>::from_vec(other.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut first_output = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut second_output = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    launch(
        map_tensors,
        (&mut device, &first, &second, &mut first_output, &mut second_output),
    )
    .await?;
    for (output, expected) in [(first_output, values), (second_output, other)] {
        assert_eq!(
            output.to_host::<m![Rows, Cols]>(&mut device.pdma).await?.into_vec(),
            expected
        );
    }
    Ok(())
}

#[tokio::test]
async fn copies_through_device_arguments() -> eyre::Result<()> {
    use furiosa_opt_examples::algebraic::{lend_device, lend_stored, pass_device};
    let mut device = Device::new(pass_device.topology())?;
    let values: Vec<i32> = (0..Rows::SIZE * Cols::SIZE).map(|i| i as i32 - 123).collect();
    let input = HostTensor::<i32, m![Rows, Cols]>::from_vec(values.clone())
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut first = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut second = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut lent = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    let mut stored = HostTensor::<i32, m![Rows, Cols]>::zero()
        .to_hbm::<m![1], m![Rows, Cols]>(&mut device.pdma)
        .await?;
    launch(pass_device, (&mut device, &input, &mut first, &mut second)).await?;
    launch(lend_device, (&mut device, &input, &mut lent)).await?;
    launch(lend_stored, (&mut device, &input, &mut stored)).await?;
    for output in [first, second, lent, stored] {
        assert_eq!(
            output.to_host::<m![Rows, Cols]>(&mut device.pdma).await?.into_vec(),
            values
        );
    }
    Ok(())
}
