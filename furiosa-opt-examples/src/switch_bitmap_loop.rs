//! A `CustomBroadcast` fill repeated in a loop: the snoop bitmap routing the switch is one constant.

use furiosa_opt_std::prelude::*;

axes![A = 64, Y = 4, R = 8, K = 32, X = 32];

type Chip = m![1];
type Cluster = m![1 # 2];

#[device(chip = 1)]
pub fn switch_bitmap_loop(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![A, R, K]>,
    identity: &HbmTensor<i8, Chip, m![X, K]>,
    out: &mut HbmTensor<i32, Chip, m![A, Y, X, R]>,
) {
    let dm: DmTensor<i8, Chip, Cluster, m![A, 1 # 4], m![R, K]> = input.to_dm(&mut device.tdma);
    let identity: DmTensor<i8, Chip, Cluster, m![A, Y], m![X, K]> = identity.to_dm(&mut device.tdma);
    for _ in 0..4 {
        let trf: TrfTensor<i8, Chip, Cluster, m![A, Y], m![R], m![K]> = device
            .sub
            .begin(dm.view())
            .fetch::<m![R], m![K]>()
            .fetch_cast::<i8>()
            .switch::<m![A, Y], m![R]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
            .collect::<m![R], m![K]>()
            .to_trf();
        let scores: DmTensor<i32, Chip, Cluster, m![A, Y], m![X, R]> = device
            .main
            .begin(identity.view())
            .fetch::<m![X], m![K]>()
            .fetch_cast::<i8>()
            .collect::<m![X], m![K]>()
            .contract_outer::<m![X], m![K], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![X]>()
            .contract_lane::<m![X], m![R]>(LaneMode::Interleaved)
            .commit_trim::<m![R]>()
            .commit();
        scores.view().to_hbm_view(&mut device.tdma, out.view_mut());
    }
}
