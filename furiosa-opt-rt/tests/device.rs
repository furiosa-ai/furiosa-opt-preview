//! Hardware tests of the runtime, each on two idle NPUs. They are `#[ignore]`d, so
//! `make furiosa-opt-test` skips them; `make furiosa-opt-hardware-test` runs them one at a time.

use std::alloc::System;
use std::sync::Arc;

use furiosa_opt_abi::arch::rngd;
use furiosa_opt_abi::executable;
use furiosa_opt_rt::memory::Kind;
use furiosa_opt_rt::{Argument, Buffer, Device, Function, Pinned, Topology};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Transfer sizes in bytes, up to and past the 16 MiB that one DMA descriptor copies.
const SIZES: [usize; 5] = [4096, 65536, 1 << 20, 16 << 20, 17 << 20];
/// Poison bytes around a transfer, so a copy that lands off target is seen.
const GUARD: usize = 256;
/// Launches per argument count in the table tests.
const SAMPLES: usize = 64;
/// Words of one cluster's record of a launch: sum, last value, argument count and cluster column.
const RECORD: usize = 4;

fn open() -> furiosa_opt_rt::Result<Arc<Device>> {
    Device::new(Topology { chips: 2, pes: 8 }).map(Arc::new)
}

/// Clusters on each chip of `device`.
fn clusters(device: &Device) -> usize {
    rngd::clusters(device.topology()) / usize::from(device.topology().chips)
}

/// `code` on every cluster, binding up to `capacity` arguments in order.
async fn load(device: &Arc<Device>, code: &[u8], capacity: usize) -> furiosa_opt_rt::Result<Function> {
    let bytes =
        executable::Executable::encode(&[vec![code; rngd::clusters(device.topology())]]).expect("fixture executable");
    // SAFETY: fixtures access only the device and mapped host operands declared by each caller.
    unsafe {
        Function::load(device, &bytes, 0, capacity, |args, bound| {
            args.enumerate()
                .try_for_each(|(index, argument)| bound.bind(index, argument))?;
            Ok(std::future::ready(Ok(())))
        })
        .await
    }
}

/// A distinct word at every 8-byte position, so a misplaced page or a stale copy cannot match.
fn fill(bytes: &mut [u8], seed: usize) {
    for (index, word) in bytes.chunks_mut(8).enumerate() {
        word.copy_from_slice(&(index as u64 + ((seed as u64) << 32)).to_le_bytes()[..word.len()]);
    }
}

/// Room for the record of every cluster of every launch of [`launches`].
fn alloc_records(device: &Device) -> furiosa_opt_rt::Result<Buffer> {
    device.alloc(Kind::Dram, SAMPLES * clusters(device) * RECORD * size_of::<u64>())
}

/// Launches `function` `SAMPLES` times. Launch `index` takes the `index`th share of `records` and
/// the values `1..count`, each offset by `index * 10000`; every third observer is dropped.
async fn launches(function: &Function, records: &mut Buffer, count: u64) -> furiosa_opt_rt::Result<()> {
    let share = records.len() / SAMPLES;
    for index in 0..SAMPLES {
        let launch = function.launch(
            std::iter::once(records.slice_mut(index * share..(index + 1) * share)?.into())
                .chain((1..count).map(|value| Argument::Value(value + index as u64 * 10000))),
        )?;
        // Launches finish in order, so awaiting the last one covers the dropped ones.
        if index % 3 == 0 && index + 1 < SAMPLES {
            drop(launch);
        } else {
            launch.await?;
        }
    }
    Ok(())
}

/// On NPUs, chips copy to their peers while host transfers progress beside the firmware's DMA.
#[tokio::test]
#[ignore = "requires two idle NPUs"]
async fn peer_transfers() -> furiosa_opt_rt::Result<()> {
    let device = open()?;
    let members = usize::from(device.topology().chips);
    let function = load(&device, include_bytes!("fixtures/pdma.bin"), 4).await?;
    for size in SIZES {
        let mut source = device.alloc(Kind::Dram, size * 2)?;
        let mut destination = device.alloc(Kind::DramShared, size * 2)?;
        let mut abandoned = device.alloc(Kind::DramShared, size * 2)?;
        let mut input = Pinned::zeroed(size * 2 * members)?;
        let mut output = Pinned::zeroed(input.len())?;
        for batch in [1, 2, 8] {
            fill(&mut input, size + batch);
            device.write([(input.slice(..), source.slice_mut(..)?)]).await?;
            drop(function.launch([
                source.slice(..)?.into(),
                abandoned.slice_mut(..)?.into(),
                Argument::Value(size as u64),
                Argument::Value(batch as u64),
            ])?);
            let launch = function.launch([
                source.slice(..)?.into(),
                destination.slice_mut(..)?.into(),
                Argument::Value(size as u64),
                Argument::Value(batch as u64),
            ])?;
            // Host transfers must progress while firmware owns its separate DMA channels.
            let mut scratch = device.alloc(Kind::Dram, size * 2)?;
            device.write([(input.slice(..), scratch.slice_mut(..)?)]).await?;
            device.read([(scratch.slice(..)?, output.slice_mut(..))]).await?;
            assert_eq!(&*input, &*output);
            launch.await?;
            for buffer in [&destination, &abandoned] {
                device.read([(buffer.slice(..)?, output.slice_mut(..))]).await?;
                for rank in 0..members {
                    let from = (rank + members - 1) % members;
                    assert_eq!(
                        &output[rank * size * 2..(rank + 1) * size * 2],
                        &input[from * size * 2..(from + 1) * size * 2],
                    );
                }
            }
        }
    }
    Ok(())
}

/// On NPUs, device code copies to and from mapped `Pinned` memory, which waits for it. Needs
/// firmware host DMA and PE DMA-map driver support.
#[tokio::test]
#[ignore = "requires two idle NPUs"]
async fn device_host_transfers() -> furiosa_opt_rt::Result<()> {
    let device = open()?;
    let members = usize::from(device.topology().chips);
    let clusters = clusters(&device);
    let function = load(&device, include_bytes!("fixtures/host_dma.bin"), 6).await?;
    for size in SIZES {
        let span = size * clusters;
        let payload = span * members;
        let mut buffer = device.alloc(Kind::Dram, span + GUARD * 2)?;
        let mut source = Pinned::zeroed(payload + GUARD * 2)?;
        let mut destination = Pinned::zeroed(source.len())?;
        let mut verify = Pinned::zeroed(buffer.len() * members)?;
        device.map(&mut source).await?;
        device.map(&mut destination).await?;
        for channels in [1, 2] {
            source.fill(0xa5);
            destination.fill(0xa5);
            verify.fill(0xa5);
            fill(&mut source[GUARD..GUARD + payload], size + channels);
            device.write([(verify.slice(..), buffer.slice_mut(..)?)]).await?;
            for write in [true, false] {
                let operands = if write {
                    [
                        Argument::Host(source.slice_mut(..)),
                        buffer.slice_mut(GUARD..GUARD + span)?.into(),
                    ]
                } else {
                    [
                        buffer.slice(GUARD..GUARD + span)?.into(),
                        Argument::Host(destination.slice_mut(..)),
                    ]
                };
                drop(function.launch(operands.into_iter().chain([
                    Argument::Value(size as u64),
                    Argument::Value(channels as u64),
                    Argument::Value(GUARD as u64),
                    Argument::Value(clusters as u64),
                ]))?);
                // Host access waits for completion even after the observer is dropped.
                if write {
                    assert_eq!(source[0], 0xa5);
                    device.read([(buffer.slice(..)?, verify.slice_mut(..))]).await?;
                    for rank in 0..members {
                        let actual = &verify[rank * buffer.len()..(rank + 1) * buffer.len()];
                        assert!(
                            actual[..GUARD]
                                .iter()
                                .chain(&actual[GUARD + span..])
                                .all(|&byte| byte == 0xa5)
                        );
                        assert_eq!(
                            &actual[GUARD..GUARD + span],
                            &source[GUARD + rank * span..GUARD + (rank + 1) * span],
                        );
                    }
                } else {
                    assert_eq!(&*source, &*destination);
                }
            }
        }
    }
    Ok(())
}

/// On NPUs, `write` and `read` of mapped `Pinned` memory run on the clusters' own channels beside a
/// launch in flight, from any offset into the mapping, and land every chip's bytes. Needs firmware
/// host DMA and PE DMA-map driver support.
#[tokio::test]
#[ignore = "requires two idle NPUs"]
async fn mapped_transfers() -> furiosa_opt_rt::Result<()> {
    let device = open()?;
    let members = usize::from(device.topology().chips);
    let function = load(&device, include_bytes!("fixtures/pdma.bin"), 4).await?;
    for size in SIZES {
        let source = device.alloc(Kind::Dram, size * 2)?;
        let mut destination = device.alloc(Kind::DramShared, size * 2)?;
        let mut buffer = device.alloc(Kind::Dram, size)?;
        let mut input = Pinned::zeroed(GUARD + size * members)?;
        let mut output = Pinned::zeroed(input.len())?;
        device.map(&mut input).await?;
        device.map(&mut output).await?;
        fill(&mut input, size);
        output.fill(0xa5);
        let launch = function.launch([
            source.slice(..)?.into(),
            destination.slice_mut(..)?.into(),
            Argument::Value(size as u64),
            Argument::Value(8),
        ])?;
        device.write([(input.slice(GUARD..), buffer.slice_mut(..)?)]).await?;
        device.read([(buffer.slice(..)?, output.slice_mut(GUARD..))]).await?;
        assert!(output[..GUARD].iter().all(|&byte| byte == 0xa5));
        assert_eq!(&input[GUARD..], &output[GUARD..]);
        launch.await?;
    }
    Ok(())
}

/// Argument counts: a few, either side of what one launch frame carries, and far past it.
const COUNTS: [u64; 4] = [3, 120, 121, 4097];

/// On NPUs, a launch past one frame reads its arguments from the table, and completes when its
/// observer is dropped. Every cluster records, per launch, the sum and last of its values (the
/// arithmetic series of `1..count` shifted by the launch's offset), their count and its column.
#[tokio::test]
#[ignore = "requires two idle NPUs"]
async fn reads_arguments_from_table() -> furiosa_opt_rt::Result<()> {
    let device = open()?;
    let (chips, clusters) = (usize::from(device.topology().chips), clusters(&device));
    let mut records = alloc_records(&device)?;
    let mut readback = Pinned::zeroed(SAMPLES * clusters * chips * RECORD * size_of::<u64>())?;
    device.write([(readback.slice(..), records.slice_mut(..)?)]).await?;
    for count in COUNTS {
        let function = load(&device, include_bytes!("fixtures/arguments.bin"), count as usize).await?;
        launches(&function, &mut records, count).await?;
        device.read([(records.slice(..)?, readback.slice_mut(..))]).await?;
        let words: &[u64] = bytemuck::cast_slice(&readback);
        for rank in 0..chips {
            for index in 0..SAMPLES {
                for cluster in 0..clusters {
                    let at = ((rank * SAMPLES + index) * clusters + cluster) * RECORD;
                    assert_eq!(
                        &words[at..at + RECORD],
                        &[
                            count * (count - 1) / 2 + (count - 1) * index as u64 * 10000,
                            count - 1 + index as u64 * 10000,
                            count,
                            (rank * clusters + cluster) as u64,
                        ],
                        "rank={rank} cluster={cluster} sample={index}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// On NPUs, launching with any number of arguments allocates nothing on any thread.
#[tokio::test]
#[ignore = "requires two idle NPUs"]
async fn launch_allocates_nothing() -> furiosa_opt_rt::Result<()> {
    let device = open()?;
    let mut records = alloc_records(&device)?;
    for count in COUNTS {
        let function = load(&device, include_bytes!("fixtures/arguments.bin"), count as usize).await?;
        // The first launches size the runtime's pools.
        launches(&function, &mut records, count).await?;
        let region = Region::new(ALLOCATOR);
        launches(&function, &mut records, count).await?;
        let stats = region.change();
        assert_eq!(
            stats.allocations + stats.reallocations,
            0,
            "allocations on all threads for {count} arguments"
        );
    }
    Ok(())
}
