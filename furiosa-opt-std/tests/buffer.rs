use furiosa_opt_std::prelude::*;

axes![A = 16];

type Word = HbmTensor<i32, m![1], m![A]>;

fn invalid<T>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::Invalid(_)))
}

/// A tensor made from a buffer reads that buffer's bytes, and gives the buffer back.
#[tokio::test]
async fn tensors_keep_their_buffers_bytes() -> Result<(), Error> {
    let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
    let bytes = (0..128).collect::<Vec<u8>>();
    let pool = HostBufferSlice::from(&bytes).to_hbm(&mut device.pdma).await?;
    let (first, second) = pool.split_at(64)?;
    let (first, second) = (Word::try_from(first)?, Word::try_from(second)?);
    for (tensor, expected) in [(second, &bytes[64..]), (first, &bytes[..64])] {
        let read = tensor.into_buffer().to_host(&mut device.pdma).await?;
        assert_eq!(&read[..], expected);
    }
    assert_eq!(Word::alloc(&device)?.into_buffer().size(), 64);
    Ok(())
}

/// A buffer becomes a tensor or view only where its extent, chips and alignment fit the type.
#[tokio::test]
async fn rejects_storage_the_type_does_not_fit() -> Result<(), Error> {
    let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
    assert!(invalid(Word::try_from(HbmBuffer::alloc(&device, 32)?)));
    assert!(invalid(Word::try_from(HbmBuffer::alloc(&device, 128)?)));
    assert!(invalid(HbmTensor::<i32, m![2], m![A]>::try_from(HbmBuffer::alloc(
        &device, 64
    )?)));
    let (_, misaligned) = HbmBuffer::alloc(&device, 66)?.split_at(2)?;
    assert!(invalid(Word::try_from(misaligned)));
    let mut buffer = HbmBuffer::alloc(&device, 96)?;
    type View<'b> = HbmTensorView<'b, i32, m![1], m![A]>;
    assert!(View::try_from(buffer.slice(.., 32..)?).is_ok());
    assert!(invalid(View::try_from(buffer.slice(.., 36..)?)));
    assert!(invalid(View::try_from(buffer.slice(.., 32..64)?)));
    assert!(invalid(HbmTensorViewMut::<i32, m![1], m![A]>::try_from(
        buffer.slice_mut(.., 2..)?
    )));
    assert!(invalid(
        HostBufferSlice::from(&[0; 95])
            .to_hbm(&mut device.pdma)
            .output(&mut buffer)
            .await
    ));
    assert!(invalid(buffer.to_host(&mut device.pdma).output(&mut [0; 97]).await));
    Ok(())
}

/// A tensor allocates only on a device whose chips its chip extent names, as a buffer becomes one.
#[test]
fn refuses_allocations_the_device_does_not_fit() -> Result<(), Error> {
    let device = Device::new(Topology { chips: 2, pes: 8 })?;
    assert!(invalid(Word::alloc(&device)));
    assert_eq!(HbmTensor::<i32, m![2], m![A]>::alloc(&device)?.into_buffer().size(), 64);
    let device = Device::new(Topology { chips: 1, pes: 8 })?;
    assert!(invalid(HbmTensor::<i32, m![2], m![A]>::alloc(&device)));
    Ok(())
}

/// A slice moves its byte range on its chips only, chip after chip in rank order, and leaves the
/// rest of the buffer as it was; a range, chip or length outside the buffer is `Error::Invalid`.
#[tokio::test]
async fn slices_move_their_chips_only() -> Result<(), Error> {
    let mut device: Device = Device::new(Topology { chips: 2, pes: 8 })?;
    let mut buffer = HostBufferSlice::from(&[0, 1, 2, 3, 4, 5, 6, 7])
        .to_hbm(&mut device.pdma)
        .await?;
    HostBufferSlice::from(&[20, 21])
        .to_hbm(&mut device.pdma)
        .output(buffer.slice_mut(1, 2..4)?)
        .await?;
    for chip in 0..2 {
        HostBufferSlice::from(&[9])
            .to_hbm(&mut device.pdma)
            .output(buffer.slice_mut(chip, 0..1)?)
            .await?;
    }
    let all = buffer.to_host(&mut device.pdma).await?;
    assert_eq!(all[..], [9, 1, 2, 3, 9, 5, 20, 21]);
    let mut column = [0; 2];
    buffer
        .slice([0, 1], 1..2)?
        .to_host(&mut device.pdma)
        .output(&mut column)
        .await?;
    assert_eq!(column, [1, 5]);
    assert!(invalid(buffer.slice(2, ..)));
    assert!(invalid(buffer.slice(.., 3..5)));
    assert!(invalid(buffer.alias(3..5)));
    assert!(invalid(
        HostBufferSlice::from(&[0; 8])
            .to_hbm(&mut device.pdma)
            .output(buffer.slice_mut(0, ..)?)
            .await
    ));
    assert!(invalid(buffer.split_at(5)));
    Ok(())
}

/// A host buffer transfers like any other host bytes, whole or in part.
#[tokio::test]
async fn host_buffers_transfer_like_other_bytes() -> Result<(), Error> {
    let mut device: Device = Device::new(Topology { chips: 2, pes: 8 })?;
    let host = HostBuffer::from(&[0, 1, 2, 3, 4, 5, 6, 7]);
    let mut buffer = host.to_hbm(&mut device.pdma).await?;
    host.slice(..4)?
        .to_hbm(&mut device.pdma)
        .output(buffer.slice_mut(1, ..)?)
        .await?;
    let mut back = HostBuffer::zero(8);
    buffer.to_host(&mut device.pdma).output(&mut back).await?;
    assert_eq!(back[..], [0, 1, 2, 3, 0, 1, 2, 3]);
    let mut part = [0; 2];
    buffer
        .slice(0, 2..)?
        .to_host(&mut device.pdma)
        .output(&mut part)
        .await?;
    assert_eq!(part, [2, 3]);
    assert!(invalid(host.slice(..9)));
    Ok(())
}

/// A host buffer of a tensor's length becomes that host tensor and comes back out unchanged; any
/// other length is refused.
#[test]
fn host_tensors_keep_their_buffers_bytes() -> Result<(), Error> {
    let bytes = (0..64).collect::<Vec<u8>>();
    let mut host = HostBuffer::zero(bytes.len());
    host.copy_from_slice(&bytes);
    let tensor = HostTensor::<i32, m![A]>::try_from(host)?;
    assert_eq!(&*tensor.into_buffer(), &bytes[..]);
    assert!(matches!(
        HostTensor::<i32, m![A]>::try_from(HostBuffer::zero(60)),
        Err(Error::Invalid(_))
    ));
    Ok(())
}

/// A batch writes every destination's selected chips and leaves all other bytes unchanged.
#[tokio::test]
async fn batches_preserve_unselected_bytes() -> Result<(), Error> {
    let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 8 })?;
    let mut first = HostBufferSlice::from(&[9; 12]).to_hbm(&mut device.pdma).await?;
    let mut alias = first.alias(..)?;
    let mut second = HostBufferSlice::from(&[8; 12]).to_hbm(&mut device.pdma).await?;
    let host = HostBuffer::from(&[1, 2, 3, 4, 5, 6, 7, 8]);
    device
        .pdma
        .to_hbm([
            (host.slice(..2)?, first.slice_mut(0, 1..3)?),
            (host.slice(2..4)?, alias.slice_mut(1, 3..5)?),
            (host.slice(4..)?, second.slice_mut(.., ..2)?),
        ])
        .await?;
    assert_eq!(
        first.to_host(&mut device.pdma).await?[..],
        [9, 1, 2, 9, 9, 9, 9, 9, 9, 3, 4, 9]
    );
    assert_eq!(
        second.to_host(&mut device.pdma).await?[..],
        [5, 6, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8]
    );
    Ok(())
}

/// A length mismatch anywhere in a batch refuses every write.
#[tokio::test]
async fn batches_reject_mismatched_lengths() -> Result<(), Error> {
    let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 8 })?;
    for mismatch in 0..3 {
        let mut first = HostBufferSlice::from(&[9; 8]).to_hbm(&mut device.pdma).await?;
        let mut alias = first.alias(..)?;
        let mut second = HostBufferSlice::from(&[8; 8]).to_hbm(&mut device.pdma).await?;
        let host = [&[1, 2][..], &[3, 4][..], &[5, 6, 7, 8][..]];
        let host = host.into_iter().enumerate().map(|(index, bytes)| {
            if index == mismatch {
                &bytes[..bytes.len() - 1]
            } else {
                bytes
            }
        });
        let result = device
            .pdma
            .to_hbm(host.zip([
                first.slice_mut(0, ..2)?,
                alias.slice_mut(1, ..2)?,
                second.slice_mut(.., ..2)?,
            ]))
            .await;
        assert!(matches!(result, Err(Error::Invalid(_))));
        assert_eq!(first.to_host(&mut device.pdma).await?[..], [9; 8]);
        assert_eq!(second.to_host(&mut device.pdma).await?[..], [8; 8]);
    }
    Ok(())
}

/// A batch containing another device's buffer refuses every write.
#[tokio::test]
async fn batches_reject_foreign_buffers() -> Result<(), Error> {
    let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 8 })?;
    let mut other = Device::<Cpu>::new(Topology { chips: 2, pes: 8 })?;
    let mut first = HostBufferSlice::from(&[9; 8]).to_hbm(&mut device.pdma).await?;
    let mut second = HostBufferSlice::from(&[8; 8]).to_hbm(&mut other.pdma).await?;
    let result = device
        .pdma
        .to_hbm([(&[1; 8], &mut first), (&[2; 8], &mut second)])
        .await;
    assert!(matches!(result, Err(Error::Device(DeviceError::ForeignBuffer))));
    assert_eq!(first.to_host(&mut device.pdma).await?[..], [9; 8]);
    assert_eq!(second.to_host(&mut other.pdma).await?[..], [8; 8]);
    Ok(())
}

/// An empty batch succeeds without allocating device memory.
#[tokio::test]
async fn batches_accept_empty_input() -> Result<(), Error> {
    let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 8 })?;
    let available = device.memory().usage().available;
    device
        .pdma
        .to_hbm(std::iter::empty::<(&[u8], &mut HbmBuffer<Cpu>)>())
        .await?;
    assert_eq!(device.memory().usage().available, available);
    Ok(())
}
