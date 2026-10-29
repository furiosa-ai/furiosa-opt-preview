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

#[device(chip = 1)]
fn typed<T>(device: &mut Device, model: &T)
where
    for<'a> &'a T: DeviceSend,
{
}

/// A function generic over a type that is not an axis loads and launches by its type key.
#[tokio::test]
async fn kernels_load_by_type() -> Result<(), Error> {
    let mut device = Device::new(typed.topology())?;
    let kernel = Loadable::<(&mut Device, &HbmTensor<f32, m![1], m![A]>)>::load(typed, &device)?;
    let model = HbmTensor::<f32, m![1], m![A]>::alloc(&device)?;
    device.launch(kernel, (&model,)).await?;
    Ok(())
}

#[device(chip = C::SIZE)]
fn sharded<C: M>(device: &mut Device, input: &HbmTensor<i32, C, m![1]>) {}

/// A kernel loaded from a generic function has the topology its argument types select.
#[test]
fn kernel_topology_follows_arguments() {
    fn topology<F: Loadable<A>, A: DeviceSend>(_: F) -> Topology {
        <Kernel<F, A> as DeviceFn<A>>::TOPOLOGY
    }
    assert_eq!(
        topology::<_, (&mut Device, &HbmTensor<i32, m![2], m![1]>)>(sharded),
        Topology { chips: 2, pes: 8 }
    );
    assert_eq!(
        topology::<_, (&mut Device, &HbmTensor<i32, m![4], m![1]>)>(sharded),
        Topology { chips: 4, pes: 8 }
    );
}

#[device(chip = 1)]
fn traced(device: &mut Device) {
    tracing::info!("executing");
}

/// Both returned and caller-owned output launches keep execution under a launch span.
#[test]
fn execution_has_launch_ancestry() -> Result<(), Error> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Trace {
        depth: AtomicUsize,
        events: Arc<AtomicUsize>,
    }
    impl tracing::Subscriber for Trace {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            assert_eq!(attributes.metadata().name(), "launch");
            assert!(attributes.metadata().fields().field("cat").is_some());
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, _: &tracing::Event<'_>) {
            assert!(self.depth.load(Ordering::Relaxed) > 0);
            self.events.fetch_add(1, Ordering::Relaxed);
        }
        fn enter(&self, _: &tracing::span::Id) {
            self.depth.fetch_add(1, Ordering::Relaxed);
        }
        fn exit(&self, _: &tracing::span::Id) {
            self.depth.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let events = Arc::new(AtomicUsize::new(0));
    let _subscriber = tracing::subscriber::set_default(Trace {
        depth: AtomicUsize::new(0),
        events: Arc::clone(&events),
    });
    let mut device = Device::new(traced.topology())?;
    drop(launch(traced, &mut device).submit());
    drop(launch(traced, &mut device).output(&mut ()).submit());
    assert_eq!(events.load(Ordering::Relaxed), 2);
    Ok(())
}

/// A submitted host transfer borrows neither the DMA context nor the bytes it moves from, so the
/// device launches on other memory while it runs, and it yields its result when awaited later.
#[tokio::test]
async fn launches_beside_submitted_transfers() -> Result<(), Error> {
    let mut device = Device::new(step.topology())?;
    let model = HbmTensor::<f32, m![1], m![A]>::alloc(&device)?;
    let mut target = HbmBuffer::alloc(&device, 32)?;
    let mut back = HostTensor::<i32, m![A]>::zero();
    let written = HostBuffer::from(&[7; 32])
        .to_hbm(&mut device.pdma)
        .output(&mut target)
        .submit();
    let staged = HostTensor::<i32, m![A]>::from_vec(vec![9; 8])
        .to_hbm::<m![1], m![A]>(&mut device.pdma)
        .submit();
    device.launch(step, (&model,)).await?;
    written.await?;
    let staged = staged.await?;
    let read = target.to_host(&mut device.pdma).submit();
    let copied = staged.to_host(&mut device.pdma).output(&mut back).submit();
    device.launch(step, (&model,)).await?;
    assert_eq!(read.await?[..], [7; 32]);
    copied.await?;
    assert_eq!(back.into_vec(), [9; 8]);
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
