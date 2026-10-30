#![cfg(not(backend = "npu"))]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

axes![A = 8];

#[device(chip = 1)]
fn forward<X: AxisName>(device: &mut Device, model: &HbmTensor<f32, m![1], m![X]>) {}

/// A device function generic over an axis infers the axis from its arguments.
#[tokio::test]
async fn infers_generic_axis_from_arguments() {
    let mut device = Device::new(forward.topology()).unwrap();
    let model = HbmTensor::<f32, m![1], m![A]>::alloc(&device).unwrap();
    launch(forward, (&mut device, &model)).await.unwrap();
}

#[device(chip = 1)]
fn step(device: &mut Device, model: &HbmTensor<f32, m![1], m![A]>) {}

/// A loaded kernel launches repeatedly, with fresh borrows each time.
#[tokio::test]
async fn kernel_launches_repeatedly() {
    let mut device = Device::new(step.topology()).unwrap();
    let kernel = step.load(&device).unwrap();
    for _ in 0..2 {
        let model = HbmTensor::<f32, m![1], m![A]>::alloc(&device).unwrap();
        device.launch(kernel, (&model,)).await.unwrap();
    }
}

/// One chip of a buffer, as a hand-written argument may name it.
struct FirstChip(HbmBuffer);

impl DeviceSend for FirstChip {
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_>) -> Result<(), Error> {
        bind(self.0.slice(0, ..)?, access)
    }
}

#[device(chip = 2)]
fn first_chip(device: &mut Device, argument: FirstChip) {}

/// A launch refuses an argument that names only some chips of its buffer, as a device binds every
/// chip of a launch's memory.
#[tokio::test]
async fn refuses_an_argument_on_some_chips() -> Result<(), Error> {
    let mut device = Device::new(first_chip.topology())?;
    let argument = FirstChip(HbmBuffer::alloc(&device, 4)?);
    let refused = device.launch(first_chip, (argument,)).await;
    assert!(matches!(refused, Err(Error::Invalid(_))), "{refused:?}");
    Ok(())
}

/// Submission returns a launch's refusal at once, and keeps an accepted launch's result for its
/// await.
#[tokio::test]
async fn submission_reports_refusal_before_completion() -> Result<(), Error> {
    let mut device = Device::new(first_chip.topology())?;
    let argument = FirstChip(HbmBuffer::alloc(&device, 4)?);
    let refused = device.launch(first_chip, (argument,)).submit().accepted();
    assert!(matches!(refused, Err(Error::Invalid(_))), "{refused:?}");
    let mut device = Device::new(step.topology())?;
    let model = HbmTensor::<f32, m![1], m![A]>::alloc(&device)?;
    device.launch(step, (&model,)).submit().accepted()?.await
}

#[device(chip = 1, pe = 1)]
fn replace(device: &mut Device, block: &mut HbmTensor<i32, m![1], m![A]>, source: &HbmTensor<i32, m![1], m![A]>) {
    *block = source.to_hbm(&mut device.tdma);
}

/// A transfer after a launch that was only submitted reads what the launch wrote, as its buffer's
/// readiness and the device's synchronization wait for the launch too.
#[tokio::test]
async fn reads_what_a_submitted_launch_writes() -> Result<(), Error> {
    let mut device = Device::new(replace.topology())?;
    let mut block = HostTensor::<i32, m![A]>::from_vec(vec![7; 8])
        .to_hbm::<m![1], m![A]>(&mut device.pdma)
        .await?;
    let source = HostTensor::<i32, m![A]>::from_vec(vec![9; 8])
        .to_hbm::<m![1], m![A]>(&mut device.pdma)
        .await?;
    drop(
        launch(replace, (&mut device, &mut block, &source))
            .submit()
            .accepted()?,
    );
    block.ready().await?;
    device.synchronize().await?;
    assert_eq!(block.to_host::<m![A]>(&mut device.pdma).await?.into_vec(), [9; 8]);
    Ok(())
}

/// A launch whose body moves memory completes under any executor, a blocking one included.
#[test]
fn launches_under_a_blocking_executor() -> Result<(), Error> {
    futures_executor::block_on(async {
        let mut device = Device::new(replace.topology())?;
        let mut block = HostTensor::<i32, m![A]>::from_vec(vec![7; 8])
            .to_hbm::<m![1], m![A]>(&mut device.pdma)
            .await?;
        let source = HostTensor::<i32, m![A]>::from_vec(vec![9; 8])
            .to_hbm::<m![1], m![A]>(&mut device.pdma)
            .await?;
        launch(replace, (&mut device, &mut block, &source)).await?;
        assert_eq!(block.to_host::<m![A]>(&mut device.pdma).await?.into_vec(), [9; 8]);
        Ok(())
    })
}
