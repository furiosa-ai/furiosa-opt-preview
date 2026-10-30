//! A submitted host transfer orders with later launches through the memory they share, and holds
//! no device context: a launch on its bytes reads what it wrote, one on other bytes runs beside it.

use std::pin::pin;
use std::task::{Context, Poll, Waker};

use furiosa_opt_examples::param::{A, B, tuple_passthrough};
use furiosa_opt_std::prelude::*;

type Input = HbmTensor<i8, m![1], m![A, B]>;

/// Host bytes a transfer moves for longer than a small launch runs.
const LARGE: usize = 512 << 20;

fn values() -> Vec<i8> {
    (0..Input::size()).map(|byte| (byte % 251) as i8).collect()
}

/// A launch on memory a submitted transfer writes waits for it and reads the bytes it wrote.
#[tokio::test]
async fn launches_read_what_a_submitted_transfer_wrote() -> eyre::Result<()> {
    let mut device = Device::new(tuple_passthrough.topology())?;
    let host = HostTensor::<i8, m![A, B]>::from_vec(values()).pinned()?;
    let mut source = Input::alloc(&device)?;
    let written = host.to_hbm(&mut device.pdma).output(&mut source).submit();
    let copied = device.launch(tuple_passthrough, ((&source,),)).await?;
    written.await?;
    assert_eq!(copied.to_host::<m![A, B]>(&mut device.pdma).await?.into_vec(), values());
    Ok(())
}

/// A launch on other memory runs while a large submitted transfer still moves.
#[tokio::test]
async fn launches_beside_a_large_submitted_transfer() -> eyre::Result<()> {
    let mut device = Device::new(tuple_passthrough.topology())?;
    let source = HostTensor::<i8, m![A, B]>::from_vec(values())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await?;
    let large = HostBuffer::zero(LARGE);
    let mut target = HbmBuffer::alloc(&device, LARGE)?;
    let mut written = pin!(large.to_hbm(&mut device.pdma).output(&mut target).submit());
    let copied = device.launch(tuple_passthrough, ((&source,),)).await?;
    let moving = match written.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(result) => result.map(|()| false)?,
        Poll::Pending => written.await.map(|()| true)?,
    };
    // The host backend copies within submission, so only a device overlaps them.
    if cfg!(backend = "npu") {
        assert!(
            large.is_pinned() && moving,
            "the launch waited for a transfer of other memory"
        );
    }
    assert_eq!(copied.to_host::<m![A, B]>(&mut device.pdma).await?.into_vec(), values());
    Ok(())
}
