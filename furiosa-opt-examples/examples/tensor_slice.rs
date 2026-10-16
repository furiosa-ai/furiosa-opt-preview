//! Compare repeated launches with a fixed-capacity tensor and a runtime-length Slice.
//! Run with `cargo furiosa-opt run --release -p furiosa-opt-examples --example tensor_slice -- gather32 25571 100`.
#![feature(register_tool, proc_macro_hygiene)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

axes![Block = 16, Width = 2048, Small = 32, Large = 256, Capacity = 32768];
type BlockTensor = HbmTensor<bf16, m![1], m![Block, Width]>;
type Pool = HbmTensor<bf16, m![1], m![Capacity, Block, Width]>;
type Index<Count> = HbmTensor<i32, m![1], m![Count]>;
type Updates<Count> = HbmTensor<bf16, m![1], m![Count, Width]>;
type Output<Count> = HbmTensor<bf16, m![1], m![Count, Block, Width]>;

trait Layout: AxisName {
    type BlockSlices: M;
    type BlockElements: M;
    type TokenSlices: M;
    type TokenElements: M;
}

impl Layout for Small {
    type BlockSlices = m![Small, Block / 2];
    type BlockElements = m![Block % 2, Width];
    type TokenSlices = m![Small, Width / 256];
    type TokenElements = m![Width % 256];
}

impl Layout for Large {
    type BlockSlices = m![Large];
    type BlockElements = m![Block, Width];
    type TokenSlices = m![Large];
    type TokenElements = m![Width];
}

type BlockValues<Count> =
    DmTensor<bf16, m![1], m![1 # 2], <Count as Layout>::BlockSlices, <Count as Layout>::BlockElements>;
type TokenValues<Count> =
    DmTensor<bf16, m![1], m![1 # 2], <Count as Layout>::TokenSlices, <Count as Layout>::TokenElements>;

#[device(chip = 1)]
fn gather<Count: Layout>(device: &mut Device, pool: &Slice<BlockTensor>, indices: &Index<Count>) -> Output<Count> {
    let values: BlockValues<Count> = pool
        .gather::<m![1], m![Block, Width]>()
        .by_byte_offsets(indices)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
fn gather_opaque<Count: Layout>(device: &mut Device, pool: &Pool, indices: &Index<Count>) -> Output<Count> {
    let values: BlockValues<Count> = pool
        .gather::<m![Capacity], m![Block, Width]>()
        .by_byte_offsets(indices)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
fn scatter<Count: Layout>(
    device: &mut Device,
    pool: &mut Slice<BlockTensor>,
    indices: &Index<Count>,
    updates: &Updates<Count>,
) {
    let values: TokenValues<Count> = updates.to_dm(&mut device.tdma);
    pool.scatter::<m![Block], m![Width]>()
        .by_byte_offsets(indices.view())
        .from_dm(&mut device.tdma, values);
}

#[device(chip = 1)]
fn scatter_opaque<Count: Layout>(
    device: &mut Device,
    pool: &mut Pool,
    indices: &Index<Count>,
    updates: &Updates<Count>,
) {
    let values: TokenValues<Count> = updates.to_dm(&mut device.tdma);
    pool.view_mut()
        .scatter::<m![Capacity, Block], m![Width]>()
        .by_byte_offsets(indices.view())
        .from_dm(&mut device.tdma, values);
}

use serde_json::json;
use std::time::Instant;

const CELLS: usize = 16 * 2048;
const WIDTH: usize = 2048;

fn contents(id: usize, phase: usize) -> Vec<bf16> {
    (0..CELLS)
        .map(|i| bf16::from_f32(((id * 17 + i * 7 + phase * 19) % 251) as f32))
        .collect()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let operation = args.get(1).map(String::as_str).unwrap_or("gather32");
    let default_count = if operation.ends_with("256") { 256 } else { 37 };
    let count = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(default_count);
    let repeats = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(100_usize);
    let allocation = args.get(4).map(String::as_str).unwrap_or("fragmented");
    assert!(matches!(allocation, "contiguous" | "fragmented" | "wide"));
    match operation {
        "gather32" | "scatter32" => measure::<Small>(operation, count, repeats, allocation).await,
        "gather256" | "scatter256" => measure::<Large>(operation, count, repeats, allocation).await,
        _ => Err("operation must be gather32, scatter32, gather256, or scatter256".into()),
    }
}

async fn measure<Count: Layout>(
    operation: &str,
    count: usize,
    repeats: usize,
    allocation: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        count <= Capacity::SIZE && count >= Count::SIZE,
        "the pool must hold every selected block"
    );
    let is_scatter = operation.starts_with("scatter");
    let mut device = Device::new(gather.topology())?;

    let start = Instant::now();
    let gather_kernel = Loadable::<(&mut Device, &Slice<BlockTensor>, &Index<Count>)>::load(gather, &device)?;
    let opaque_gather = Loadable::<(&mut Device, &Pool, &Index<Count>)>::load(gather_opaque, &device)?;
    let scatter_kernel =
        Loadable::<(&mut Device, &mut Slice<BlockTensor>, &Index<Count>, &Updates<Count>)>::load(scatter, &device)?;
    let opaque_scatter =
        Loadable::<(&mut Device, &mut Pool, &Index<Count>, &Updates<Count>)>::load(scatter_opaque, &device)?;
    write_block.load(&device)?;
    read_block.load(&device)?;
    println!(
        "{}",
        json!({"phase": "load", "operation": operation, "ns": start.elapsed().as_nanos()})
    );

    let start = Instant::now();
    let mut opaque = Pool::alloc(&device)?;
    let mut values = Vec::with_capacity(count);
    let mut gaps = Vec::new();
    let mut wide_gap = None;
    for i in 0..count {
        if i == count / 2 && allocation == "wide" {
            wide_gap = Some(HbmTensor::<u8, m![1], m![4294967296]>::alloc(&device)?);
        }
        if i % 7 == 0 && allocation != "contiguous" {
            gaps.push(HbmTensor::<u8, m![1], m![512]>::alloc(&device)?);
        }
        values.push(BlockTensor::alloc(&device)?);
    }
    let mut blocks = Slice::from(values);
    let mut output = Output::<Count>::alloc(&device)?;
    let mut updates = Updates::<Count>::alloc(&device)?;
    let mut indices = Index::<Count>::alloc(&device)?;
    println!(
        "{}",
        json!({"phase": "allocate", "operation": operation, "blocks": count,
        "ns": start.elapsed().as_nanos(), "payload_host_dma": false})
    );

    let mut identities: Vec<_> = (0..count).collect();
    let mut selected: Vec<_> = (0..Count::SIZE).map(|i| i * (count - 1) / (Count::SIZE - 1)).collect();
    selected.rotate_left(7);

    for phase in 0..3 {
        let change = ["initial", "swap", "replacement"][phase];
        let start = Instant::now();
        if phase == 1 {
            blocks.swap(0, count - 1);
            identities.swap(0, count - 1);
        } else if phase == 2 {
            blocks[0] = BlockTensor::alloc(&device)?;
            identities[0] = count + 1;
        }
        println!(
            "{}",
            json!({"phase": "membership", "change": change,
            "operation": operation, "blocks": count, "ns": start.elapsed().as_nanos()})
        );

        selected.rotate_left(phase * 3);
        let slots: Vec<_> = (0..Count::SIZE).map(|i| (i * 7 + phase * 3) % 16).collect();
        let offsets: Vec<_> = selected
            .iter()
            .enumerate()
            .map(|(i, &block)| {
                i32::try_from((block * CELLS + if is_scatter { slots[i] * WIDTH } else { 0 }) * 2).unwrap()
            })
            .collect();
        let update: Vec<_> = (0..Count::SIZE * WIDTH)
            .map(|i| bf16::from_f32(((i * 13 + phase * 29 + 3) % 251) as f32))
            .collect();
        let start = Instant::now();
        HostTensor::<i32, m![Count]>::from_vec(offsets)
            .to_hbm(&mut device.pdma)
            .output(&mut indices)
            .await?;
        HostTensor::<bf16, m![Count, Width]>::from_vec(update.clone())
            .to_hbm(&mut device.pdma)
            .output(&mut updates)
            .await?;
        for &index in &selected {
            let data = contents(identities[index], phase);
            write(&mut device, &mut opaque, index, data.clone()).await?;
            HostTensor::<bf16, m![Block, Width]>::from_vec(data)
                .to_hbm(&mut device.pdma)
                .output(&mut blocks[index])
                .await?;
        }
        println!(
            "{}",
            json!({"phase": "upload", "change": change,
            "operation": operation, "blocks": count, "ns": start.elapsed().as_nanos()})
        );

        macro_rules! call {
            ($mode:expr) => {
                match (is_scatter, $mode) {
                    (false, "opaque") => {
                        launch(opaque_gather, (&mut device, &opaque, &indices))
                            .output(&mut output)
                            .await?
                    }
                    (false, "owned") => {
                        launch(gather_kernel, (&mut device, &blocks, &indices))
                            .output(&mut output)
                            .await?
                    }
                    (true, "opaque") => launch(opaque_scatter, (&mut device, &mut opaque, &indices, &updates)).await?,
                    (true, "owned") => launch(scatter_kernel, (&mut device, &mut blocks, &indices, &updates)).await?,
                    _ => unreachable!(),
                }
            };
        }

        for mode in ["opaque", "owned"] {
            if is_scatter {
                let start = Instant::now();
                for &index in &selected {
                    let data = contents(identities[index], phase);
                    if mode == "opaque" {
                        write(&mut device, &mut opaque, index, data).await?;
                    } else {
                        HostTensor::<bf16, m![Block, Width]>::from_vec(data)
                            .to_hbm(&mut device.pdma)
                            .output(&mut blocks[index])
                            .await?;
                    }
                }
                println!(
                    "{}",
                    json!({"phase": "reset", "change": change, "mode": mode,
                    "operation": operation, "blocks": count, "ns": start.elapsed().as_nanos()})
                );
            }
            #[cfg(backend = "npu")]
            let before = device.memory().usage().available;
            let start = Instant::now();
            call!(mode);
            println!(
                "{}",
                json!({"phase": "first_call", "change": change, "mode": mode,
                "operation": operation, "blocks": count, "ns": start.elapsed().as_nanos()})
            );

            #[cfg(backend = "npu")]
            {
                let retained = before.saturating_sub(device.memory().usage().available);
                if phase == 0 && mode == "owned" && allocation == "contiguous" {
                    assert!(retained <= CELLS * 2, "metadata exceeds one block of headroom");
                }
                println!(
                    "{}",
                    json!({"phase": "metadata", "change": change, "mode": mode,
                    "operation": operation, "blocks": count, "retained_bytes": retained})
                );
            }

            let start = Instant::now();
            if is_scatter {
                for (i, &index) in selected.iter().enumerate() {
                    let actual = if mode == "opaque" {
                        read(&mut device, &opaque, index).await?
                    } else {
                        blocks[index]
                            .to_host::<m![Block, Width]>(&mut device.pdma)
                            .await?
                            .into_vec()
                    };
                    let mut expected = contents(identities[index], phase);
                    expected[slots[i] * WIDTH..(slots[i] + 1) * WIDTH]
                        .copy_from_slice(&update[i * WIDTH..(i + 1) * WIDTH]);
                    assert_eq!(actual, expected, "scatter {change} {mode} block={index}");
                }
            } else {
                let actual = output
                    .to_host::<m![Count, Block, Width]>(&mut device.pdma)
                    .await?
                    .into_vec();
                let expected: Vec<_> = selected
                    .iter()
                    .flat_map(|&index| contents(identities[index], phase))
                    .collect();
                assert_eq!(actual, expected, "gather {change} {mode}");
            }
            println!(
                "{}",
                json!({"phase": "check", "change": change, "mode": mode,
                "operation": operation, "blocks": count, "exact": true,
                "ns": start.elapsed().as_nanos()})
            );
        }

        for round in 0..3 {
            let mut modes = ["opaque", "owned"];
            modes.rotate_left(round % 2);
            for mode in modes {
                for _ in 0..5 {
                    call!(mode);
                }
                let mut samples = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    let start = Instant::now();
                    call!(mode);
                    samples.push(start.elapsed().as_nanos());
                }
                println!(
                    "{}",
                    json!({"phase": "hot", "change": change, "round": round,
                    "mode": mode, "operation": operation, "blocks": count, "ns": samples})
                );
            }
        }
    }
    drop(gaps);
    drop(wide_gap);
    Ok(())
}

#[device(chip = 1)]
fn write_block(device: &mut Device, pool: &mut Pool, index: &HbmScalar<usize>, block: &BlockTensor) {
    let at = index.to_spm(&mut device.tdma);
    block.view().to_hbm_view(
        &mut device.tdma,
        pool.view_mut()
            .tile::<m![Capacity], 1, m![Capacity = 1 #{!} Capacity, Block, Width]>(at),
    );
}

#[device(chip = 1)]
fn read_block(device: &mut Device, pool: &Pool, index: &HbmScalar<usize>) -> BlockTensor {
    let at = index.to_spm(&mut device.tdma);
    let mut output = BlockTensor::new();
    pool.view()
        .tile::<m![Capacity], 1, m![Capacity = 1 # Capacity, Block, Width]>(at)
        .to_hbm_view(&mut device.tdma, output.view_mut());
    output
}

async fn write(
    device: &mut Device,
    pool: &mut Pool,
    index: usize,
    values: Vec<bf16>,
) -> Result<(), furiosa_opt_std::Error> {
    let index = HbmScalar::from_host(index, &mut device.pdma).await?;
    let block = HostTensor::<bf16, m![Block, Width]>::from_vec(values)
        .to_hbm(&mut device.pdma)
        .await?;
    launch(write_block, (device, pool, &index, &block)).await
}

async fn read(device: &mut Device, pool: &Pool, index: usize) -> Result<Vec<bf16>, furiosa_opt_std::Error> {
    let index = HbmScalar::from_host(index, &mut device.pdma).await?;
    let block = launch(read_block, (&mut *device, pool, &index)).await?;
    Ok(block.to_host::<m![Block, Width]>(&mut device.pdma).await?.into_vec())
}
