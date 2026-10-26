# furiosa-opt-rt

Runs compiled `#[device]` functions on Furiosa NPUs.
`furiosa-opt-std` calls it for you; use it directly to drive images from your own host code.

```rust,no_run
use furiosa_opt_rt::{Argument, Device, Pinned, Result, Topology, memory::Kind, registry};
use std::sync::Arc;

async fn run(entry: &registry::Entry) -> Result<()> {
    let device = Arc::new(Device::new(Topology { chips: 1, pes: 8 })?);
    let mut buffer = device.alloc(Kind::Dram, 4096)?;
    let mut host = Pinned::zeroed(4096)?;
    device.write([(host.slice(..), buffer.slice_mut(..)?)]).await?;
    // SAFETY: the entry holds the compiler's image of its function.
    let function = unsafe { registry::load(entry, &device, None) }.await?;
    function.launch([Argument::from(buffer.slice_mut(..)?)])?.await?;
    device.read([(buffer.slice(..)?, host.slice_mut(..))]).await
}
```

## Guarantees

- **Submitted work runs to completion.** Dropping or forgetting the future of a launch or a transfer
  cancels nothing; its memory stays held until the device answers.
- **Conflicting work waits or is refused, never races.** A launch that needs bytes other work holds
  fails with `Error::Busy`; a transfer waits for the work admitted before it on its bytes. A launch
  is admitted when it is called, a transfer when it is first polled. Launches on one device run in
  submission order.
- **A device failure is sticky.** Once submitted work fails on the device, the device refuses all
  later work with that error, and every allocation the failed work could have written is
  `Error::Poisoned`. A failure answer does not say whether device code ran, so even a refused
  launch counts. Work refused before it is submitted, such as `Error::Busy`, leaves the device as
  it was.
- **Launching allocates nothing on the host.** A model step is hundreds of launches of a few
  microseconds each, so the runtime's cost per launch is the latency floor.

## Layers

Each layer talks only to its neighbours, and a name means the same thing on the host and in the
firmware.

| Layer | Contract | Host | Firmware |
|---|---|---|---|
| Program | `#[device]` fn, `launch(f, args)` | `furiosa-opt-std` | |
| Image | compiler image, registry entry | `furiosa-opt-abi` | |
| Runtime | `Device`, `Buffer`, `Pinned`, `Function` | this crate | |
| Application | calls and their answers | `Device` | `Server` |
| Mailbox | matches each answer to its call | `Mailbox` | `Mailbox` |
| Message | `Request`, `Answer`, tagged options | `furiosa-opt-ipc` | `furiosa-opt-ipc` |
| Frame | header and words; its revision | `furiosa-opt-ipc` | `furiosa-opt-ipc` |
| Link | rings and doorbells | `Link` | `Link` |
| Device ABI | executable, bootloader, ring layout | `furiosa-opt-abi` | `furiosa-opt-abi` |

The firmware ships as a prebuilt image this crate embeds; a signed bootloader verifies and starts
it on each PE cluster.

## Kernel driver

Transfers of pinned memory mapped to a device, and DMA issued by device code, need a kernel driver
that reserves PDMA channels for the firmware. On a driver without that support the device still
opens: every host transfer takes the driver's own channels, and a launch that needs firmware DMA
fails with `Error::Unsupported`.
