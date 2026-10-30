//! Compilation fixtures for sub-context Cast and Transpose Engines.
//!
//! These kernels exercise both engines in sub while the other examples use main. Their stage-equivalence
//! tests compare outputs across VISA, LIR, and EDF.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];

axes![S = 256, R = 8, KV = 8, A = 8, B = 16];

/// Narrows `i32` to `i8` in the sub context.
#[device(chip = 1)]
pub fn sub_cast(device: &mut Device, input: &HbmTensor<i32, Chip, m![S, R, KV]>) -> HbmTensor<i8, Chip, m![S, R, KV]> {
    let dm: DmTensor<i32, Chip, Cluster, m![S], m![R, KV]> = input.to_dm(&mut device.tdma);

    let out: DmTensor<i8, Chip, Cluster, m![S], m![R, KV]> = device
        .sub
        .begin(dm.view())
        .fetch::<m![R], m![KV]>()
        .fetch_cast::<i32>()
        .collect::<m![R], m![KV]>()
        .cast::<i8, m![KV # 32]>()
        .commit_trim::<m![KV]>()
        .commit();

    out.to_hbm(&mut device.tdma)
}

/// Transposes `[A, B]` to `[B, A]` per slice in the sub context.
#[device(chip = 1)]
pub fn sub_transpose(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![S, A, B]>,
) -> HbmTensor<i8, Chip, m![S, B, A]> {
    let dm: DmTensor<i8, Chip, Cluster, m![S], m![A, B]> = input.to_dm(&mut device.tdma);

    let out: DmTensor<i8, Chip, Cluster, m![S], m![B, A]> = device
        .sub
        .begin(dm.view())
        .fetch::<m![A], m![B]>()
        .collect::<m![A], m![B # 32]>()
        .transpose::<m![B], m![A # 32]>()
        .commit_trim::<m![A]>()
        .commit();

    out.to_hbm(&mut device.tdma)
}
