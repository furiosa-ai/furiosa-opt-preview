//! Device memory moves the same way on either backend: a host transfer lays a tensor out as its
//! type says, and a launch takes the memory its arguments and returns name.

use furiosa_opt_examples::param::{A, B, tuple_passthrough};
use furiosa_opt_examples::pe_count::{P4, four_pe_add};
use furiosa_opt_std::prelude::*;

axes![R = 4, C = 3];

type Input = HbmTensor<i8, m![1], m![A, B]>;

fn foreign<T>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::Device(DeviceError::ForeignBuffer)))
}

/// A host transfer lays the tensor out as its HBM type says, and reads it back the same way.
#[tokio::test]
async fn host_transfers_lay_tensors_out_as_their_types_say() -> eyre::Result<()> {
    let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
    let values = (0..<m![R, C]>::SIZE as i32).collect::<Vec<_>>();
    let hbm = HostTensor::<i32, m![R, C]>::from_vec(values.clone())
        .to_hbm::<m![1], m![C, R]>(&mut device.pdma)
        .await?;
    let buffer = hbm.into_buffer();
    let bytes = buffer.to_host(&mut device.pdma).await?;
    let columns = (0..C::SIZE).flat_map(|column| (0..R::SIZE).map(move |row| (row * C::SIZE + column) as i32));
    assert_eq!(Vec::from(bytes), columns.flat_map(i32::to_le_bytes).collect::<Vec<_>>());
    let hbm = HbmTensor::<i32, m![1], m![C, R]>::try_from(buffer)?;
    assert_eq!(hbm.to_host::<m![R, C]>(&mut device.pdma).await?.into_vec(), values);
    Ok(())
}

/// A launch writes the memory its returns name, so every owner of that memory sees the write.
#[tokio::test]
async fn writes_reach_every_owner_of_the_memory() -> eyre::Result<()> {
    let mut device = Device::new(tuple_passthrough.topology())?;
    let bytes = (0..Input::size()).map(|byte| byte as u8).collect::<Vec<_>>();
    let source = HostBufferSlice::from(&bytes).to_hbm(&mut device.pdma).await?;
    let source = Input::try_from(source)?;
    let buffer = HbmBuffer::alloc(&device, bytes.len())?;
    let alias = buffer.alias(..)?;
    let mut target = Input::try_from(buffer)?;
    launch(tuple_passthrough, (&mut device, (&source,)))
        .output(&mut target)
        .await?;
    let read = alias.to_host(&mut device.pdma).await?;
    assert_eq!(read[..], bytes[..]);
    Ok(())
}

/// A launch whose write overlaps memory it also reads through another owner is refused, as the
/// device refuses it.
#[tokio::test]
async fn refuses_a_launch_that_writes_memory_it_reads() -> eyre::Result<()> {
    let mut device = Device::new(tuple_passthrough.topology())?;
    let buffer = HbmBuffer::alloc(&device, Input::size())?;
    let source = Input::try_from(buffer.alias(..)?)?;
    let mut target = Input::try_from(buffer)?;
    let refused = launch(tuple_passthrough, (&mut device, (&source,)))
        .output(&mut target)
        .await;
    assert!(matches!(refused, Err(Error::Device(DeviceError::Busy))), "{refused:?}");
    Ok(())
}

/// A transfer or launch on one device refuses memory another device allocated.
#[tokio::test]
async fn refuses_memory_of_another_device() -> eyre::Result<()> {
    type Source = HbmTensor<i32, m![1], m![P4]>;
    let owner = Device::new(four_pe_add.topology())?;
    // The other half of the owner's chip, so a one-chip host holds both devices.
    let mut other = Device::new(Placement {
        pes: 4..8,
        ..owner.placement().clone()
    })?;
    let mut buffer = HbmBuffer::alloc(&owner, Source::size())?;
    let zeros = vec![0; Source::size()];
    assert!(foreign(
        HostBufferSlice::from(&zeros)
            .to_hbm(&mut other.pdma)
            .output(&mut buffer)
            .await
    ));
    assert!(foreign(buffer.to_host(&mut other.pdma).await));
    let source = Source::try_from(buffer)?;
    assert!(foreign(launch(four_pe_add, (&mut other, &source)).await));
    Ok(())
}
