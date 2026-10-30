#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#![cfg(not(backend = "npu"))]

use furiosa_opt_std::prelude::*;

axes![Width = 512];
type Buffer<D> = HbmTensor<D, m![1], m![Width]>;
type Values<D> = DmTensor<D, m![1], m![1], m![Width / 8], m![Width % 8]>;

#[device(chip = 1)]
fn scale(device: &mut Device, scores: &Buffer<f32>, mask: &Buffer<i32>) -> Buffer<f32> {
    let mask: Values<i32> = mask.to_dm(&mut device.tdma);
    let log = device
        .main
        .begin(mask.view())
        .fetch::<m![1], m![Width % 8]>()
        .collect::<m![1], m![Width % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Equal(0i32),
            Cmp::True,
            Cmp::False,
            Cmp::True,
        ]))
        .vector_log::<m![Width % 8]>(&mut device.sub);
    let scores: Values<f32> = scores.to_dm(&mut device.tdma);
    let output: Values<f32> = device
        .main
        .begin(scores.view())
        .fetch::<m![1], m![Width % 8]>()
        .collect::<m![1], m![Width % 8]>()
        .vector_init()
        .vector_tagged(&log)
        .vector_scale(Branched::imm(
            TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]),
            0.5f32,
        ))
        .vector_final()
        .commit_trim::<m![Width % 8]>()
        .commit();
    output.to_hbm(&mut device.tdma)
}

/// Branch-log scaling selects a predicate, not one exact execution ID.
#[tokio::test]
async fn scales_selected_ids() {
    let mut device = Device::new(scale.topology()).expect("CPU device");
    let scores =
        HostTensor::<f32, m![Width]>::from_vec([2., 4., 6., 8., 10., 12., 14., 16.].repeat(<m![Width / 8]>::SIZE))
            .to_hbm::<m![1], m![Width]>(&mut device.pdma)
            .await
            .expect("scores");
    let mask = HostTensor::<i32, m![Width]>::from_vec([0, 1, -1, i32::MAX, 0, 2, -2, 7].repeat(<m![Width / 8]>::SIZE))
        .to_hbm::<m![1], m![Width]>(&mut device.pdma)
        .await
        .expect("mask");
    let result = launch(scale, (&mut device, &scores, &mask)).await.expect("CPU scale");
    assert_eq!(
        result
            .to_host::<m![Width]>(&mut device.pdma)
            .await
            .expect("result")
            .into_vec(),
        [2., 2., 3., 4., 10., 6., 7., 8.].repeat(<m![Width / 8]>::SIZE),
    );
}
