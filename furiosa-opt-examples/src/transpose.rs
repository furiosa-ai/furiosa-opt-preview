//! Transpose examples.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1];
axes![P = 64, A = 8, B = 16, C = 32, D = 4, E = 8, F = 2, A4 = 16, B4 = 32];

/// Device function that transposes a tensor from shape [A, B, C] to [C, A, B].
/// it is divided into two steps for demonstration purposes.
#[device(chip = 1)]
pub fn transpose_simple(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![A, B, C]>,
) -> HbmTensor<f32, Chip, m![C, A, B]> {
    // transpose: [A, B, C] -> [A, C, B]
    let intermediate: HbmTensor<f32, Chip, m![A, C, B]> = input.to_hbm(&mut device.tdma);

    // transpose: [A, C, B] -> [C, A, B]
    intermediate.to_hbm(&mut device.tdma)
}

/// Transposes an i8 tensor from `[P, A, B]` to `[P, B, A]` with the TU
/// transpose engine.
///
/// The tensor is staged in SRAM as `[P=64] [A=8, B=16]`, then streamed as:
///
/// ```text
/// collect:   [A=8]  [B=16 # 32]
/// transpose: [B=16] [A=8  # 32]
/// commit:    [B=16] [A=8]
/// ```
#[device(chip = 1, pe = 1)]
pub fn transpose_i8_tu(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![P, A, B]>,
) -> HbmTensor<i8, Chip, m![P, B, A]> {
    let input_dm: DmTensor<i8, Chip, Cluster, m![P], m![A, B]> = input.to_dm(&mut device.tdma);

    let output_dm: DmTensor<i8, Chip, Cluster, m![P], m![B, A]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![B]>()
        .collect::<m![A], m![B # 32]>()
        .transpose::<m![B], m![A # 32]>()
        .commit_trim::<m![A]>()
        .commit();

    output_dm.to_hbm(&mut device.tdma)
}

/// Transposes an i16 tensor with doubled input packing.
#[device(chip = 1, pe = 1)]
pub fn transpose_i16_tu(
    device: &mut Device,
    input: &HbmTensor<i16, Chip, m![P, D, B]>,
) -> HbmTensor<i16, Chip, m![P, B, D]> {
    let input_dm: DmTensor<i16, Chip, Cluster, m![P], m![D, B]> = input.to_dm(&mut device.tdma);

    let output_dm: DmTensor<i16, Chip, Cluster, m![P], m![B, D]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![D], m![B]>()
        .collect::<m![D], m![B]>()
        .transpose::<m![B], m![D # 16]>()
        .commit_trim::<m![D]>()
        .commit();

    output_dm.to_hbm(&mut device.tdma)
}

/// Transposes an `i4` tensor from `[P, A4, B4]` to `[P, B4, A4]`, with the
/// 16 input rows a 4-bit element allows (16 x 4 bits = the 8-byte row width).
#[device(chip = 1, pe = 1)]
pub fn transpose_i4_tu(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, m![P, A4, B4]>,
) -> HbmTensor<i4, Chip, m![P, B4, A4]> {
    let input_dm: DmTensor<i4, Chip, Cluster, m![P], m![A4, B4]> = input.to_dm(&mut device.tdma);

    let output_dm: DmTensor<i4, Chip, Cluster, m![P], m![B4, A4]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![A4, B4 / 16], m![B4 % 16]>()
        .collect::<m![A4, B4 / 16], m![B4 % 16 # 64]>()
        .transpose::<m![B4], m![A4 # 64]>()
        .commit_trim::<m![A4]>()
        .commit();

    output_dm.to_hbm(&mut device.tdma)
}

/// Transposes an `f32` tensor from `[P, F, E]` to `[P, E, F]` and commits it as `bf16`.
///
/// The Transpose Engine sees the `f32` stream; the commit unit narrows it to `bf16` after the
/// transpose, so the engine must be configured for 32-bit elements, not for the committed type. A
/// 32-bit stream feeds the engine at most two rows.
#[device(chip = 1, pe = 1)]
pub fn transpose_f32_commit_cast_tu(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![P, F, E]>,
) -> HbmTensor<bf16, Chip, m![P, E, F]> {
    let input_dm: DmTensor<f32, Chip, Cluster, m![P], m![F, E]> = input.to_dm(&mut device.tdma);

    // A converting commit reads at least 16 B per packet, so the two-row packet keeps its padding.
    let output_dm: DmTensor<bf16, Chip, Cluster, m![P], m![E, F # 4]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![F], m![E]>()
        .collect::<m![F], m![E]>()
        .transpose::<m![E], m![F # 8]>()
        .commit_trim::<m![F # 4]>()
        .commit_cast::<bf16>()
        .commit();

    output_dm.to_hbm(&mut device.tdma)
}
