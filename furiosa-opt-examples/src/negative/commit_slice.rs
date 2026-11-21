//! Rejection fixtures for the slices a commit writes.

use furiosa_opt_std::prelude::*;

axes![A = 8, B = 8, O = 8];

type Chip = m![1];
type Cluster = m![1];
type Slices = m![A, B];

/// The tile's other slices hold no defined value, and the destination would take them as data.
#[device(chip = 1, pe = 1)]
pub fn undefined_slice_committed_to_a_valid_slice(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, O]>,
    previous: &HbmTensor<i32, Chip, m![A, B, O]>,
) -> HbmTensor<i32, Chip, m![A, B, O]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![O]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, Chip, Cluster, Slices, m![O]> = previous.to_dm(&mut device.tdma);
    let source = source.view().slice_tile::<m![B / 4], 1, m![A, B % 4 # 8]>(0);
    device
        .main
        .begin(source)
        .fetch::<m![1], m![O]>()
        .collect::<m![1], m![O]>()
        .commit_trim::<m![O]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}

/// A commit writes each slice in place, so it cannot carry slice `[a, b]` to slice `[b, a]`.
#[device(chip = 1, pe = 1)]
pub fn slice_moved_by_a_commit(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, O]>,
    previous: &HbmTensor<i32, Chip, m![B, A, O]>,
) -> HbmTensor<i32, Chip, m![B, A, O]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![O]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, Chip, Cluster, m![B, A], m![O]> = previous.to_dm(&mut device.tdma);
    device
        .main
        .begin(source.view())
        .fetch::<m![1], m![O]>()
        .collect::<m![1], m![O]>()
        .commit_trim::<m![O]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}
