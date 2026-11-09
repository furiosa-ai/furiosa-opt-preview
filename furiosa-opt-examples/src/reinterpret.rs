use furiosa_opt_std::prelude::*;

axes![S = 512, R = 8, W = 32];

/// One landing of each slice's `R` rows of `W` bytes, read back as FP32 twice: every row, and the
/// last half alone, as a weight's scales sit behind its codes.
#[device(chip = 1)]
pub fn reinterpret(
    device: &mut Device,
    bytes: &HbmTensor<i8, m![1], m![S, R, W]>,
    halves: &mut HbmTensor<f32, m![1], m![S, R, W / 4]>,
) -> HbmTensor<f32, m![1], m![S, R, W / 4]> {
    let landed: DmTensor<i8, m![1], m![S / 256], m![S % 256], m![R, W]> = bytes.to_dm(&mut device.tdma);
    // SAFETY: each FP32 cell is the four bytes it sits on, in order.
    let words = unsafe { landed.view().reinterpret::<f32, m![R, W / 4]>() };
    let scales = words.clone().tile::<m![R / 4], 1, m![R / 4 = 1 # 2, R % 4, W / 4]>(1);
    scales.to_hbm_view(
        &mut device.tdma,
        halves
            .view_mut()
            .tile::<m![R / 4], 1, m![S, R / 4 = 1 #{!} 2, R % 4, W / 4]>(1),
    );
    let mut whole: DmTensor<f32, m![1], m![S / 256], m![S % 256], m![R, W / 4]> = DmTensor::new();
    words.to_dm_view(&mut device.tdma, whole.view_mut());
    whole.to_hbm(&mut device.tdma)
}
