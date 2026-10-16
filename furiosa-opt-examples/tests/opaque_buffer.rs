#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

#[derive(Debug, Clone)]
struct Width;

impl AxisName for Width {
    const NAME: Ident = Ident::D;
    const SIZE: usize = 64;
}

type Tile = Symbol<Width>;
type Values = HbmTensor<bf16, Identity, Tile>;
type Local = DmTensor<bf16, Identity, Padding<Identity, Broadcast<2>>, Padding<Identity, Broadcast<256>>, Tile>;

#[device(chip = 1)]
fn read(device: &mut Device, buffer: &HbmBuffer, offset: &HbmScalar<usize>) -> Values {
    let offset = offset.to_spm(&mut device.tdma);
    let tile: Local = HbmTensorView::<bf16, Identity, Tile>::new(buffer, offset).to_dm(&mut device.tdma);
    tile.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
fn write(device: &mut Device, buffer: &mut HbmBuffer, offset: &HbmScalar<usize>, values: &Values) {
    let offset = offset.to_spm(&mut device.tdma);
    let tile: Local = values.to_dm(&mut device.tdma);
    tile.view().to_hbm_view(
        &mut device.tdma,
        HbmTensorViewMut::<bf16, Identity, Tile>::new(buffer, offset),
    );
}

#[derive(Debug, Clone)]
struct Lane;

impl AxisName for Lane {
    const NAME: Ident = Ident::K;
    const SIZE: usize = 16;
}

#[derive(Debug, Clone)]
struct Selected;

impl AxisName for Selected {
    const NAME: Ident = Ident::G;
    const SIZE: usize = 32;
}

type Payload = Symbol<Lane>;
type Selection = Symbol<Selected>;
type Indices = HbmTensor<i32, Identity, Selection>;
type Gathered = HbmTensor<bf16, Identity, Pair<Selection, Payload>>;
type Sparse = DmTensor<bf16, Identity, Padding<Identity, Broadcast<2>>, Padding<Selection, Broadcast<256>>, Payload>;
type Spatial = DmTensor<
    bf16,
    Identity,
    Padding<Identity, Broadcast<2>>,
    Padding<Stride<Payload, Broadcast<4>>, Broadcast<256>>,
    Pair<Selection, Modulo<Payload, Broadcast<4>>>,
>;

#[device(chip = 1)]
fn gather(device: &mut Device, buffer: &HbmBuffer, indices: &Indices) -> Gathered {
    let values: Sparse = HbmTensorView::<bf16, Identity, Payload>::gather_buffer(buffer)
        .by_byte_offsets(indices)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
fn scatter(device: &mut Device, buffer: &mut HbmBuffer, indices: &Indices, values: &Gathered) {
    let values: Sparse = values.to_dm(&mut device.tdma);
    HbmTensorViewMut::<bf16, Identity, Payload>::scatter_buffer(buffer)
        .by_byte_offsets(indices.view())
        .from_dm(&mut device.tdma, values);
}

// HBM stores contiguous payloads even when their lanes occupy different SRAM slices.
#[device(chip = 1)]
fn scatter_spatial(device: &mut Device, buffer: &mut HbmBuffer, indices: &Indices, values: &Gathered) {
    let values: Spatial = values.to_dm(&mut device.tdma);
    HbmTensorViewMut::<bf16, Identity, Payload>::scatter_buffer(buffer)
        .by_byte_offsets(indices.view())
        .from_dm(&mut device.tdma, values);
}

#[derive(Debug, Clone)]
struct Storage;

impl AxisName for Storage {
    const NAME: Ident = Ident::R;
    const SIZE: usize = 10880;
}

type StaticPool = HbmTensor<bf16, Identity, Pair<Symbol<Storage>, Payload>>;

#[device(chip = 1)]
fn gather_static(device: &mut Device, buffer: &StaticPool, indices: &Indices) -> Gathered {
    let values: Sparse = buffer
        .gather::<Symbol<Storage>, Payload>()
        .by_byte_offsets(indices)
        .to_dm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[tokio::test(flavor = "current_thread")]
async fn gathers_fixed_payloads_by_byte_offset() -> eyre::Result<()> {
    let mut device = Device::new(gather_static.topology())?;
    let values: Vec<_> = (0..Storage::SIZE * Lane::SIZE)
        .map(|at| bf16::from_bits((at % 65536) as u16))
        .collect();
    let buffer = HostTensor::<bf16, Pair<Symbol<Storage>, Payload>>::from_vec(values.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let offsets: Vec<_> = (0..Selected::SIZE)
        .map(|i| ((Storage::SIZE - 1) * i / (Selected::SIZE - 1) * Lane::SIZE * 2) as i32)
        .collect();
    let indices = HostTensor::<i32, Selection>::from_vec(offsets.clone())
        .to_hbm(&mut device.pdma)
        .await?;

    let actual = launch(gather_static, (&mut device, &buffer, &indices))
        .await?
        .to_host::<Pair<Selection, Payload>>(&mut device.pdma)
        .await?
        .into_vec();

    let expected: Vec<_> = offsets
        .iter()
        .flat_map(|&offset| {
            let start = usize::try_from(offset).unwrap() / 2;
            values[start..start + Lane::SIZE].iter().map(|value| value.to_bits())
        })
        .collect();
    assert_eq!(
        actual.into_iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        expected
    );
    Ok(())
}

// Native allocations have no CPU staging bytes. Check their contents against a host oracle.
#[cfg(backend = "npu")]
#[tokio::test(flavor = "current_thread")]
async fn accesses_actual_capacity() -> eyre::Result<()> {
    let mut device = Device::new(read.topology())?;
    let replacement: Vec<_> = (0..64).map(|value| bf16::from_f32(value as f32 + 1.0)).collect();
    let values = HostTensor::<bf16, Tile>::from_vec(replacement.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    for (blocks, block_size, heads, head_dim) in [(1, 1, 1, 64), (7, 3, 3, 80), (13, 8, 4, 128), (17, 4, 8, 160)] {
        let bytes = blocks * block_size * 2 * heads * head_dim * 2;
        let mut buffer = HbmBuffer::alloc(&device, bytes)?;
        let mut expected: Vec<u8> = (0..bytes).map(|at| (at * 17 + at / 251) as u8).collect();
        HostBufferSlice::from(&expected)
            .to_hbm(&mut device.pdma)
            .output(&mut buffer)
            .await?;
        for at in [0, bytes / 2, bytes - 128] {
            let offset = HbmScalar::from_host(at, &mut device.pdma).await?;
            let output = launch(read, (&mut device, &buffer, &offset)).await?;
            let actual: Vec<_> = output
                .to_host::<Tile>(&mut device.pdma)
                .await?
                .into_vec()
                .into_iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect();
            assert_eq!(actual, expected[at..at + 128]);
            launch(write, (&mut device, &mut buffer, &offset, &values)).await?;
            for (target, value) in expected[at..at + 128].chunks_exact_mut(2).zip(&replacement) {
                target.copy_from_slice(&value.to_bits().to_le_bytes());
            }
            let actual = buffer.to_host(&mut device.pdma).await?;
            assert_eq!(actual[..], expected[..], "unselected bytes changed");
        }
        for at in [1, bytes - 126, usize::MAX - 1] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    HbmTensorView::<bf16, Identity, Tile>::new(&buffer, at)
                }))
                .is_err()
            );
        }
    }

    let replacement: Vec<_> = (0..512).map(|value| bf16::from_f32(value as f32 + 0.5)).collect();
    let values = HostTensor::<bf16, Pair<Selection, Payload>>::from_vec(replacement.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    for bytes in [1024, 20160, 212992, 348160] {
        let mut buffer = HbmBuffer::alloc(&device, bytes)?;
        let mut expected: Vec<u8> = (0..bytes).map(|at| (at * 17 + at / 251) as u8).collect();
        HostBufferSlice::from(&expected)
            .to_hbm(&mut device.pdma)
            .output(&mut buffer)
            .await?;
        let offsets: Vec<_> = (0..32).map(|i| ((bytes / 32 - 1) * i / 31 * 32) as i32).collect();
        let indices = HostTensor::<i32, Selection>::from_vec(offsets.clone())
            .to_hbm(&mut device.pdma)
            .await?;
        let actual = launch(gather, (&mut device, &buffer, &indices))
            .await?
            .to_host::<Pair<Selection, Payload>>(&mut device.pdma)
            .await?
            .into_vec();
        let oracle: Vec<_> = offsets
            .iter()
            .flat_map(|&at| expected[at as usize..at as usize + 32].iter().copied())
            .collect();
        assert_eq!(
            actual
                .into_iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
            oracle
        );
        launch(scatter, (&mut device, &mut buffer, &indices, &values)).await?;
        launch(scatter_spatial, (&mut device, &mut buffer, &indices, &values)).await?;
        for (&at, payload) in offsets.iter().zip(replacement.chunks_exact(16)) {
            for (target, value) in expected[at as usize..at as usize + 32].chunks_exact_mut(2).zip(payload) {
                target.copy_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        let actual = buffer.to_host(&mut device.pdma).await?;
        assert_eq!(actual[..], expected[..], "scatter changed unselected bytes");
        if bytes == 348160 {
            let reference = HostTensor::<bf16, Pair<Symbol<Storage>, Payload>>::from_vec(
                expected
                    .chunks_exact(2)
                    .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]]))),
            )
            .to_hbm(&mut device.pdma)
            .await?;
            let mut timings = [Vec::new(), Vec::new()];
            for round in 0..120 {
                for path in [round % 2, 1 - round % 2] {
                    let start = std::time::Instant::now();
                    let result = if path == 0 {
                        launch(gather_static, (&mut device, &reference, &indices)).await?
                    } else {
                        launch(gather, (&mut device, &buffer, &indices)).await?
                    };
                    drop(result);
                    if round >= 20 {
                        timings[path].push(start.elapsed().as_nanos());
                    }
                }
            }
            for values in &mut timings {
                values.sort_unstable();
            }
            eprintln!(
                "gather launch ns: static p50={} p95={}, buffer p50={} p95={}",
                timings[0][50], timings[0][95], timings[1][50], timings[1][95]
            );
        }
    }

    let kernel = read.load(&device)?;
    let offset = HbmScalar::from_host(0_usize, &mut device.pdma).await?;
    let mut output = Values::alloc(&device)?;
    let available = device.memory().usage().available;
    let mut buffer = HbmBuffer::alloc(&device, 1024)?;
    HostBufferSlice::from(&[0_u8; 1024])
        .to_hbm(&mut device.pdma)
        .output(&mut buffer)
        .await?;
    launch(kernel, (&mut device, &buffer, &offset))
        .output(&mut output)
        .await?;
    let prepared = device.memory().usage().available;
    launch(kernel, (&mut device, &buffer, &offset))
        .output(&mut output)
        .await?;
    assert_eq!(device.memory().usage().available, prepared);
    drop(buffer);
    assert_eq!(
        device.memory().usage().available,
        available,
        "buffer metadata retained an ownership cycle"
    );

    let buffer = HbmBuffer::alloc(&device, 256)?;
    let offset = HbmScalar::from_host(130usize, &mut device.pdma).await?;
    assert!(
        launch(read, (&mut device, &buffer, &offset)).await.is_err(),
        "device view exceeded the actual allocation"
    );
    Ok(())
}
