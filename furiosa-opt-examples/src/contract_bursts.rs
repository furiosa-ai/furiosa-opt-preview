//! Bursts of short contractions, each burst's result exchanged between the two clusters.
//!
//! - Each contraction is a throttled Tensor Unit operation: a burst makes many clock-control
//!   requests back to back, and the exchange keeps each cluster waiting for the other.

use furiosa_opt_std::prelude::*;

axes![A = 8, K = 32, R = 8, P = 2];

type Chip = m![1];
type Cluster = m![P];
type Slice = m![1 # 256];
type Output = HbmTensor<i32, Chip, m![P, A, R]>;

/// Contractions in one burst.
const BURST: usize = 64;

/// Two bursts of `input` contracted with `weight`; each output is the other cluster's burst.
#[device(chip = 1)]
pub fn contract_bursts(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![P, A, K]>,
    weight: &HbmTensor<i8, Chip, m![P, R, K]>,
) -> (Output, Output) {
    let input = input.to_dm::<Cluster, Slice, m![A, K]>(&mut device.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![R, K]>(&mut device.tdma);
    let trf: TrfTensor<i8, Chip, Cluster, Slice, m![R], m![K]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![R], m![K]>()
        .fetch_cast::<i8>()
        .collect::<m![R], m![K]>()
        .to_trf();
    let first = burst(device, &input, &trf);
    let second = burst(device, &input, &trf);
    (first.to_hbm(&mut device.tdma), second.to_hbm(&mut device.tdma))
}

/// `BURST` contractions into one tensor, which the other cluster then receives.
fn burst(
    device: &mut Device,
    input: &DmTensor<i8, Chip, Cluster, Slice, m![A, K]>,
    trf: &TrfTensor<i8, Chip, Cluster, Slice, m![R], m![K]>,
) -> DmTensor<i32, Chip, Cluster, Slice, m![A, R]> {
    let mut local: DmTensor<i32, Chip, Cluster, Slice, m![A, R]> = DmTensor::new();
    for _ in 0..BURST {
        device
            .main
            .begin(input.view())
            .fetch::<m![A], m![K]>()
            .fetch_cast::<i8>()
            .collect::<m![A], m![K]>()
            .contract_outer::<m![A], m![K], _, _, _>(trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![A]>()
            .contract_lane::<m![A], m![R]>(LaneMode::Interleaved)
            .commit_trim::<m![R]>()
            .commit_view(local.view_mut());
    }
    local.cluster_swap().to_dm::<m![A, R]>(&mut device.tdma)
}
