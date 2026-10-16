# furiosa-opt-rt

Runs compiled device functions from the host: opens a device, owns its memory, and launches
functions on it. `furiosa-opt-std` generates these calls for `#[device]` functions.

```rust,no_run
use furiosa_opt_rt::{Argument, Device, Result, registry};
use std::sync::Arc;

async fn run<'a>(
    image: &registry::Image,
    device: Arc<Device>,
    arguments: impl IntoIterator<Item = Argument<'a>>,
) -> Result<()> {
    image(device, None).await?.launch(arguments)?.await
}
```

## Devices and launches

- `Device::new` takes a `Topology`, which opens the first free placement of that shape, or a
  `Placement`, which opens exactly there; `device.placement()` tells where it opened.
  Topology candidates use the chips `FURIOSA_VISIBLE_CHIPS` lists; an explicit placement uses its
  chip ranks and PE range unchanged. Opening fails with `Error::Reserved` while other processes hold
  all. Each device keeps a worker thread that spins on a core while its launches or transfers run
  and wakes every millisecond while idle.
- `launch` submits before it returns, so submit several launches before awaiting them. A launch
  keeps its arguments until the device finishes, even when its Future is dropped or forgotten.
- The last drop of a device, its functions and its launches waits for the work the device
  admitted and closes it, so the placement is free for the next `Device::new`. A launch's epilogue
  may hold its device, which then closes after the epilogue ends. A failed epilogue fails only its
  own launch; failed hardware fails every later one.
- A program loads only with the SDK, Rust toolchain and target it was built with.
- Overlapping launches are refused unless both functions opt into [`Function::ordered`].
  Its unsafe caller obligation is that every cluster finishes all caller-memory accesses before
  any cluster answers a launch, including indirect memory and host DMA.
  The compiler's exit DRAM drain and cluster rendezvous provide that ordering within each chip;
  firmware arrival order alone does not, and the rendezvous does not cover cross-chip accesses.
  Compiler images opt in for chip-local DRAM and SRAM. Multi-chip `DramShared` stays exclusive,
  and generic `Function::load` tasks stay exclusive unless their unsafe caller proves the ordering.
  Exhausted launch identities fail with `Error::Capacity` rather than wrapping.
- `make furiosa-opt-hardware-test` runs the `#[ignore]`d hardware tests of this crate and the
  compiler's loader, one at a time, on two idle NPUs.

## Memory

```rust,no_run
# async fn transfer(device: &furiosa_opt_rt::Device) -> furiosa_opt_rt::Result<()> {
    use furiosa_opt_rt::{Pinned, memory::Kind};
    let mut buffer = device.alloc(Kind::Dram, 4096)?;
    let mut bytes = Pinned::zeroed(4096 * usize::from(device.topology().chips))?;
    device.write([(bytes.slice(..), buffer.slice_mut(..)?)]).await?;
    device.read([(buffer.slice(..)?, bytes.slice_mut(..))]).await?;
#   Ok(())
# }
```

- A `Buffer` owns one byte range at the same offset on every chip: `slice` and `slice_mut` borrow
  it, `split_at` divides it into two owners, and nothing clones it. `alias` makes a second owner
  of a range; a conflicting access through either owner is `Error::Busy`, except between distinct
  ordered launches on the same device using chip-local memory; multi-chip `DramShared` stays exclusive.
- A write to device bytes a launch or transfer still uses is `Error::Busy` until the device
  finishes, except between distinct ordered launches using chip-local memory; reads and disjoint ranges overlap.
  Pinned host memory is held as a whole: ordered host operands share only the same device mapping,
  excluding transfers and unordered launches. Two host operands of one launch remain `Error::Busy`.
- Ordered launches retain every access until hardware completion, so transfers and user leases
  still conflict with them. Aliases within one launch remain excluded, and failures still poison memory.
- A transfer covers the selected chips in rank order and needs an exact host length.
- `Device::map` registers `Pinned` memory for the device's own DMA: device code copies to and
  from it, taking it as `Argument::Host`, and `write` and `read` of it run on the clusters'
  channels beside the launches in flight, except write batches of at most 2 KiB, which go through
  the BAR4 window. Other host memory moves through the kernel driver's.
- A kernel driver that cannot reserve PDMA channels for the firmware leaves the device no DMA of
  its own: `map` and a launch's `Argument::Host` fail with `Error::Device`, `write` and `read`
  move pinned memory on the driver's channels, and device code that moves host or peer memory
  fails its launch.
- Memory held by work whose outcome is unknown after a failure is `Error::Poisoned`.
