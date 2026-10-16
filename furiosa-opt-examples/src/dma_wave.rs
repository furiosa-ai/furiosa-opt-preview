//! Lands one wave of a row permutation: the source view leaves out the wave digit, the
//! destination tile trims it, and both hold the same periodic rows.

use furiosa_opt_std::prelude::*;

axes![R = 64, H = 64];

type Owner = DmTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![R / 32, R % 32, H]>;

#[device(chip = 1)]
pub fn wave(
    device: &mut Device,
    input: &HbmTensor<bf16, m![1], m![R, H]>,
) -> HbmTensor<bf16, m![1], m![R / 32, R % 32, H]> {
    let mut owner: Owner = DmTensor::new();
    // SAFETY: memory row `16 a + x` of the first 32 rows is row `32 a + x`.
    let rows = unsafe {
        input
            .view()
            .tile::<m![R], 32, m![R = 32 # R, H]>(0)
            .reshape::<m![1], m![R / 32 # 4, R % 16, H]>()
    };
    rows.to_dm_view(
        &mut device.tdma,
        owner
            .view_mut()
            .tile::<m![R % 32], 16, m![R / 32, R % 32 = 16 #{!} 32, H]>(0),
    );
    let mut output = HbmTensor::new();
    owner
        .view()
        .tile::<m![R % 32], 16, m![R / 32, R % 32 = 16 # 32, H]>(0)
        .to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![R % 32], 16, m![R / 32, R % 32 = 16 #{!} 32, H]>(0),
        );
    output
}
