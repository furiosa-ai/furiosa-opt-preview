//! Launching a loaded kernel allocates nothing on the host. Gated to the `npu` backend, since the
//! host backend runs each kernel's Rust body, which allocates:
//!
//!   cargo furiosa-opt test -p furiosa-opt-examples --test allocations
#![cfg(backend = "npu")]

use std::alloc::System;

use furiosa_opt_examples::param::{A, B, MANY, explicit_unit, long_array_copy};
use furiosa_opt_ipc::MAX_LAUNCH_ARGS;
use furiosa_opt_std::prelude::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

// `long_array_copy`'s inputs alone exceed one launch frame, so its launches go through the table.
const _: () = assert!(MANY > MAX_LAUNCH_ARGS);

const LAUNCHES: usize = 64;

type Hbm = HbmTensor<i8, m![1], m![A, B]>;

/// Launches allocate nothing on any thread once their kernels are loaded, whether the arguments
/// fit one launch frame or go in a table that changes with every launch. One test, since the
/// count spans every thread.
#[tokio::test]
async fn launch_allocates_nothing() -> eyre::Result<()> {
    let mut device = Device::new(explicit_unit.topology())?;
    let inputs = (0..=MANY).map(|_| Hbm::alloc(&device)).collect::<Result<Vec<_>, _>>()?;
    let windows: [&[Hbm; MANY]; 2] = [inputs[..MANY].try_into()?, inputs[1..].try_into()?];
    let mut output = Hbm::alloc(&device)?;
    let mut launches = async || -> eyre::Result<()> {
        for launch in 0..LAUNCHES {
            device.launch(explicit_unit, (&inputs[0], &mut output)).await?;
            device
                .launch(long_array_copy, (windows[launch % 2], &mut output))
                .await?;
        }
        Ok(())
    };
    // The first round loads both kernels and sizes the runtime's pools.
    launches().await?;
    let region = Region::new(ALLOCATOR);
    launches().await?;
    let change = region.change();
    assert_eq!(change.allocations + change.reallocations, 0, "launches allocated");
    Ok(())
}
