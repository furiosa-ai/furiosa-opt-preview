#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#![cfg(not(backend = "npu"))]

use furiosa_opt_std::prelude::*;

axes![One = 1];
type Pool = Slice<Block>;
type Block = HbmTensor<i32, m![1], m![One]>;

#[device(chip = 1, pe = 1)]
fn read<Length: AxisName>(
    device: &mut Device,
    blocks: &Slice<HbmTensor<i32, m![1], m![Length]>>,
) -> HbmTensor<i32, m![1], m![Length]> {
    blocks[0].to_hbm(&mut device.tdma)
}

#[device(chip = 1, pe = 1)]
fn read_pool(device: &mut Device, blocks: &Pool) -> Block {
    blocks[0].to_hbm(&mut device.tdma)
}

#[device(chip = 1, pe = 1)]
fn replace(device: &mut Device, blocks: &mut Pool, source: &Block) {
    blocks[0] = source.to_hbm(&mut device.tdma);
}

#[tokio::test]
async fn launches_owned_collections() {
    let mut device = Device::new(read.topology()).unwrap();
    let block = HostTensor::<i32, m![One]>::from_vec(vec![7])
        .to_hbm::<m![1], m![One]>(&mut device.pdma)
        .await
        .unwrap();
    let mut blocks = Slice::from(vec![block]);
    let result = launch(read, (&mut device, &blocks)).await.unwrap();
    assert_eq!(
        result.to_host::<m![One]>(&mut device.pdma).await.unwrap().into_vec(),
        [7]
    );
    let result = launch(read_pool, (&mut device, &blocks)).await.unwrap();
    assert_eq!(
        result.to_host::<m![One]>(&mut device.pdma).await.unwrap().into_vec(),
        [7]
    );

    let source = HostTensor::<i32, m![One]>::from_vec(vec![9])
        .to_hbm::<m![1], m![One]>(&mut device.pdma)
        .await
        .unwrap();
    launch(replace, (&mut device, &mut blocks, &source)).await.unwrap();
    assert_eq!(
        blocks[0].to_host::<m![One]>(&mut device.pdma).await.unwrap().into_vec(),
        [9]
    );

    let array = [source];
    blocks.extend(array);
    let result = launch(read_pool, (&mut device, &blocks)).await.unwrap();
    assert_eq!(
        result.to_host::<m![One]>(&mut device.pdma).await.unwrap().into_vec(),
        [9]
    );

    blocks.swap(0, 1);
    blocks.truncate(1);
    let result = launch(read, (&mut device, &blocks)).await.unwrap();
    assert_eq!(
        result.to_host::<m![One]>(&mut device.pdma).await.unwrap().into_vec(),
        [9]
    );
}
