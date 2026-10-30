#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

type Result<A, B, C> = (A, B, C);
struct Ok;

axes![Row = 4];

type Staged = DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Row]>;

#[derive(DeviceSend)]
struct Parameters {
    input: HbmTensor<i32, m![1], m![Row]>,
}

#[device(chip = 1)]
fn copy(device: &mut Device, parameters: &Parameters) -> HbmTensor<i32, m![1], m![Row]> {
    let input: Staged = parameters.input.view().to_dm(&mut device.tdma);
    input.to_hbm(&mut device.tdma)
}

#[tokio::test]
async fn supports_shadowed_result_names() {
    let _: Result<_, _, _> = (1, 2, 3);
    let _ = Ok;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).expect("device");
    let parameters = Parameters {
        input: HostTensor::<i32, m![Row]>::from_vec([3, 1, 4, 1])
            .to_hbm(&mut device.pdma)
            .await
            .expect("upload"),
    };

    let mut output = launch(copy, (&mut device, &parameters)).await.expect("launch");
    assert_eq!(
        output
            .to_host::<m![Row]>(&mut device.pdma)
            .await
            .expect("download")
            .into_vec(),
        [3, 1, 4, 1]
    );

    let parameters = Parameters {
        input: HostTensor::<i32, m![Row]>::from_vec([-2, 7, 0, 42])
            .to_hbm(&mut device.pdma)
            .await
            .expect("second upload"),
    };
    launch(copy, (&mut device, &parameters))
        .output(&mut output)
        .await
        .expect("output launch");
    assert_eq!(
        output
            .to_host::<m![Row]>(&mut device.pdma)
            .await
            .expect("download")
            .into_vec(),
        [-2, 7, 0, 42]
    );
}
