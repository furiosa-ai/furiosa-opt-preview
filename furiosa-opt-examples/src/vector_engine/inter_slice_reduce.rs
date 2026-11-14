//! Inter-slice reductions with reordered promotions, residual time axes, and broadcasts.

use furiosa_opt_std::prelude::*;

use super::{I, P, Q, R, R8, T, U, V, W, X};

type Chip = m![1];
type Cluster = m![1 # 2];

/// Two time axes promoted together, in their time order.
#[device(chip = 1)]
pub fn inter_slice_reduce_promote_two(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![W, R, I, Q, P]>,
) -> HbmTensor<f32, Chip, m![W, I, Q, P]> {
    let input_dm = input.to_dm::<Cluster, m![W, R], m![I, Q, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![W, I, Q], m![P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![I, Q], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![I, Q], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![W, I, Q], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// Two promoted axes landing in the opposite order from their time order.
#[device(chip = 1)]
pub fn inter_slice_reduce_promote_swapped(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![W, R, I, Q, P]>,
) -> HbmTensor<f32, Chip, m![W, Q, I, P]> {
    let input_dm = input.to_dm::<Cluster, m![W, R], m![I, Q, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![W, Q, I], m![P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![I, Q], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![I, Q], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![W, Q, I], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// One time axis promoted while `U` stays in time.
#[device(chip = 1)]
pub fn inter_slice_reduce_time_residue(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![W, R, T, U, P]>,
) -> HbmTensor<f32, Chip, m![W, T, U, P]> {
    let input_dm = input.to_dm::<Cluster, m![W, R], m![T, U, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![W, T], m![U, P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![T, U], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![T, U], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![W, T], m![U]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// Non-adjacent promotion: `I` and `U` go to slice while `Q` between them stays in time.
#[device(chip = 1)]
pub fn inter_slice_reduce_promote_skipping(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![W, R, I, Q, U, P]>,
) -> HbmTensor<f32, Chip, m![W, I, U, Q, P]> {
    let input_dm = input.to_dm::<Cluster, m![W, R], m![I, Q, U, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![W, I, U], m![Q, P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![I, Q, U], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![I, Q, U], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![W, I, U], m![Q]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// Three time axes promoted at once.
#[device(chip = 1)]
pub fn inter_slice_reduce_promote_three(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![V, R8, I, Q, U, P]>,
) -> HbmTensor<f32, Chip, m![V, I, Q, U, P]> {
    let input_dm = input.to_dm::<Cluster, m![V, R8], m![I, Q, U, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![V, I, Q, U], m![P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![I, Q, U], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![I, Q, U], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![V, I, Q, U], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// A broadcast position sitting between two promoted axes.
#[device(chip = 1)]
pub fn inter_slice_reduce_broadcast_between(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![V, R8, I, U, P]>,
) -> HbmTensor<f32, Chip, m![V, I, Q, U, P]> {
    let input_dm = input.to_dm::<Cluster, m![V, R8], m![I, U, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![V, I, 1 # 2, U], m![P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![I, U], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![I, U], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![V, I, 1 # 2, U], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// A broadcast position ahead of the promoted axes.
#[device(chip = 1)]
pub fn inter_slice_reduce_broadcast_leading(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![V, R8, I, U, P]>,
) -> HbmTensor<f32, Chip, m![V, Q, I, U, P]> {
    let input_dm = input.to_dm::<Cluster, m![V, R8], m![I, U, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![V, 1 # 2, I, U], m![P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![I, U], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![I, U], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![V, 1 # 2, I, U], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// The smallest reduce group: two slices collapse into one promoted axis of two.
#[device(chip = 1)]
pub fn inter_slice_reduce_group_two(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![X, I, Q, P]>,
) -> HbmTensor<f32, Chip, m![X, Q, P]> {
    let input_dm = input.to_dm::<Cluster, m![X, I], m![Q, P]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![X, Q], m![P]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![Q], m![P]>()
        .fetch_cast::<f32>()
        .collect::<m![Q], m![P]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![X, Q], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![P]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}
