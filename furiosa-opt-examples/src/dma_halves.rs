//! Copies a tensor through the two halves of one DM tensor, one DMA per half and direction.
//!
//! The two loads write disjoint halves, so neither needs to wait for the other.

use furiosa_opt_std::prelude::*;

axes![R = 128, Half = 2, H = 128];

type Owner = DmTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![Half, R, H]>;

#[device(chip = 1)]
pub fn halves(
    device: &mut Device,
    input: &HbmTensor<bf16, m![1], m![Half, R, H]>,
) -> HbmTensor<bf16, m![1], m![Half, R, H]> {
    let mut owner: Owner = DmTensor::new();
    input.view().tile::<m![Half], 1, m![Half = 1 # 2, R, H]>(0).to_dm_view(
        &mut device.tdma,
        owner.view_mut().tile::<m![Half], 1, m![Half = 1 #{!} 2, R, H]>(0),
    );
    input.view().tile::<m![Half], 1, m![Half = 1 # 2, R, H]>(1).to_dm_view(
        &mut device.tdma,
        owner.view_mut().tile::<m![Half], 1, m![Half = 1 #{!} 2, R, H]>(1),
    );
    let mut output = HbmTensor::new();
    owner.view().tile::<m![Half], 1, m![Half = 1 # 2, R, H]>(0).to_hbm_view(
        &mut device.tdma,
        output.view_mut().tile::<m![Half], 1, m![Half = 1 #{!} 2, R, H]>(0),
    );
    owner.view().tile::<m![Half], 1, m![Half = 1 # 2, R, H]>(1).to_hbm_view(
        &mut device.tdma,
        output.view_mut().tile::<m![Half], 1, m![Half = 1 #{!} 2, R, H]>(1),
    );
    output
}
