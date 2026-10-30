//! Rows of one host buffer move apart: typed as tensors, they move to and from device tensors in
//! either order, and the CPU reaches one row while a transfer still moves another.

use std::pin::pin;
use std::task::{Context, Poll, Waker};

use furiosa_opt_std::prelude::*;

axes![Head = 4, Lane = 64];

type Row = m![Head, Lane];
type Block = HbmTensor<i32, m![1], m![Lane, Head]>;

/// Host bytes a transfer moves for longer than a CPU access to other bytes takes.
const LARGE: usize = 512 << 20;

/// Rows of a host buffer, typed as tensors, move to and from device tensors as host tensors of
/// those types do, relaid out where a row's order is not the device's.
#[tokio::test]
async fn views_move_rows_of_host_memory() -> eyre::Result<()> {
    let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
    let mut host = HostBuffer::zero(3 * Block::size());
    let first = host.slice_mut(..)?.chunks(Block::size()).next().expect("a first row");
    first
        .into_bytes()?
        .copy_from_slice(&HostTensor::<i32, Row>::from_vec(0..Row::SIZE as i32).into_buffer());
    let row = |index: usize| index * Block::size()..(index + 1) * Block::size();

    let tensor = HostTensorView::<i32, Row>::try_from(host.slice(row(0))?)?
        .to_hbm::<m![1], m![Lane, Head]>(&mut device.pdma)
        .await?;
    (tensor.to_host::<Row>(&mut device.pdma))
        .output(HostTensorViewMut::try_from(host.slice_mut(row(1))?)?)
        .await?;
    (tensor.to_host::<m![Lane, Head]>(&mut device.pdma))
        .output(HostTensorViewMut::try_from(host.slice_mut(row(2))?)?)
        .await?;
    let mut copy = Block::alloc(&device)?;
    (HostTensorView::<i32, m![Lane, Head]>::try_from(host.slice(row(2))?)?.to_hbm(&mut device.pdma))
        .output(&mut copy)
        .await?;

    let lane_major =
        (0..Lane::SIZE).flat_map(|lane| (0..Head::SIZE).map(move |head| (head * Lane::SIZE + lane) as i32));
    let expected = HostTensor::<i32, m![Lane, Head]>::from_vec(lane_major).into_buffer();
    assert_eq!(host[row(1)], host[row(0)]);
    assert_eq!(host[row(2)], expected[..]);
    assert_eq!(
        copy.to_host::<Row>(&mut device.pdma).await?.into_buffer()[..],
        host[row(0)]
    );
    if cfg!(backend = "npu") {
        assert!(host.is_pinned(), "the rows moved through pageable memory");
    }
    Ok(())
}

/// The CPU reads one row of a host buffer while a transfer still writes the others.
#[tokio::test]
async fn reads_a_row_beside_a_moving_transfer() -> eyre::Result<()> {
    let mut device: Device = Device::new(Topology { chips: 1, pes: 8 })?;
    let source = HbmBuffer::alloc(&device, LARGE)?;
    let mut host = HostBuffer::zero(Block::size() + LARGE);
    let (first, rest) = host.slice_mut(..)?.split_at(Block::size())?;
    first.into_bytes()?.fill(7);
    let mut read = pin!(source.to_host(&mut device.pdma).output(rest).submit());
    let row = host.slice(..Block::size())?.bytes()?.to_vec();
    let moving = match read.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(result) => result.map(|()| false)?,
        Poll::Pending => read.await.map(|()| true)?,
    };
    assert_eq!(row, vec![7; Block::size()]);
    // The host backend copies within submission, so only a device overlaps them.
    if cfg!(backend = "npu") {
        assert!(
            host.is_pinned() && moving,
            "the CPU read waited for a transfer of other bytes"
        );
    }
    Ok(())
}

/// Blocks picked from a slice, in any order, move in one batch as the memory they alias.
#[tokio::test]
async fn picked_blocks_move_in_one_batch() -> eyre::Result<()> {
    let mut device: Device = Device::new(Topology { chips: 1, pes: 8 })?;
    let row = |index: usize| index * Block::size()..(index + 1) * Block::size();
    let bytes = (0..4 * Block::size()).map(|byte| byte as u8).collect::<Vec<_>>();
    let pool = HostBufferSlice::from(&bytes).to_hbm(&mut device.pdma).await?;
    let blocks = (0..4)
        .map(|index| pool.alias(row(index)).and_then(Block::try_from))
        .collect::<Result<Slice<_>, Error>>()?;
    // SAFETY: each transfer below completes before the next one starts.
    let mut picked = unsafe { blocks.pick(&[3, 1]) }?;

    let mut host = HostBuffer::zero(2 * Block::size());
    let rows = host.slice_mut(..)?.chunks(Block::size());
    device.pdma.to_host(picked.iter().zip(rows)).await?;
    assert_eq!(host[row(0)], bytes[row(3)]);
    assert_eq!(host[row(1)], bytes[row(1)]);

    let rows = host.slice(..)?.chunks(Block::size());
    device.pdma.to_hbm(rows.zip(picked.iter_mut().rev())).await?;
    let swapped = [row(0), row(3), row(2), row(1)].map(|range| &bytes[range]).concat();
    assert_eq!(pool.to_host(&mut device.pdma).await?[..], swapped[..]);
    Ok(())
}
