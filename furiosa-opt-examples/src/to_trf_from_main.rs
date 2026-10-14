//! Compilation fixture for a main-context TRF store followed by contraction.
//!
//! The store takes `&mut device.sub` because its write occupies sub as well as main. This exercises
//! the main-context store path; the other TRF examples issue their stores directly from sub.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];

axes![S = 256, R = 8, K = 32, X = 32];

/// Stages `weight` into the TRF from the main context, then contracts `input` against it.
#[device(chip = 1)]
pub fn contract_trf_from_main(
    device: &mut Device,
    weight: &HbmTensor<i8, Chip, m![S, R, K]>,
    input: &HbmTensor<i8, Chip, m![X, K]>,
) -> HbmTensor<i32, Chip, m![S, X, R]> {
    let weight_dm: DmTensor<i8, Chip, Cluster, m![S], m![R, K]> = weight.to_dm(&mut device.tdma);
    let input_dm: DmTensor<i8, Chip, Cluster, m![S], m![X, K]> = input.to_dm(&mut device.tdma);

    let trf: TrfTensor<i8, Chip, Cluster, m![S], m![R], m![K]> = device
        .main
        .begin(weight_dm.view())
        .fetch::<m![R], m![K]>()
        .fetch_cast::<i8>()
        .collect::<m![R], m![K]>()
        .to_trf(&mut device.sub);

    let out: DmTensor<i32, Chip, Cluster, m![S], m![X, R]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![X], m![K]>()
        .fetch_cast::<i8>()
        .collect::<m![X], m![K]>()
        .contract_outer::<m![X], m![K], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![X]>()
        .contract_lane::<m![X], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit();

    out.to_hbm(&mut device.tdma)
}
