use super::*;

/// Stores a cast-only VE output through the sub context.
#[device(chip = 1)]
pub fn ve_cast_only_to_vrf(device: &mut Device, input: &HbmTensor<i32, Chip, m![A]>) {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut device.tdma);
    let _vrf: VrfTensor<i4, Chip, Cluster, m![A / 2], m![A % 2 # 64]> = device
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_cast::<i4, m![A % 2 # 64]>()
        .vector_final()
        .to_vrf();
}

/// Stores an arithmetic and cast result into a VRF view through the main context.
#[device(chip = 1)]
pub fn ve_cast_arithmetic_to_vrf(device: &mut Device, input: &HbmTensor<i32, Chip, m![A]>) {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut device.tdma);
    let mut vrf = VrfTensor::<i4, Chip, Cluster, m![A / 2], m![A % 2 # 64]>::new();
    device
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 7)
        .vector_cast::<i4, m![A % 2 # 64]>()
        .vector_final()
        .to_vrf_view(&mut device.sub, vrf.view_mut());
}
