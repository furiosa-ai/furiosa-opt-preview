# furiosa-opt-rt

Runs compiled device functions on Furiosa NPUs from the host.
`furiosa-opt-std` generates these calls for `#[device]` functions.

```rust,no_run
use furiosa_opt_rt::{Argument, Device, Pinned, Result, Topology, memory::Kind, registry};
use std::sync::Arc;

async fn run(entry: &registry::Entry) -> Result<()> {
    let device = Arc::new(Device::new(Topology { chips: 1, pes: 8 })?);
    let mut buffer = device.alloc(Kind::Dram, 4096)?;
    let mut host = Pinned::zeroed(4096)?;
    device.write([(host.slice(..), buffer.slice_mut(..)?)]).await?;
    // SAFETY: the entry holds the compiler's image of its function.
    let function = unsafe { entry.load(&device, None) }.await?;
    function.launch([Argument::from(buffer.slice_mut(..)?)])?.await?;
    device.read([(buffer.slice(..)?, host.slice_mut(..))]).await
}
```

- A `Device` is the chips and PEs one process reserved; dropping its last handle waits for its work and frees them.
- A `Buffer` is one byte range at the same offset on every chip of its device.
- A `Function` is a loaded image; `launch` submits at once and returns a future of its completion.
- Work that overlaps memory other running work writes is refused with `Error::Busy`, and once any launch or transfer fails, the device refuses all later work with that error.
