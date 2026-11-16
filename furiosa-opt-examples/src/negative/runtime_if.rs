//! Rejection fixtures for runtime `if` expressions that violate compile-time requirements.
//! Instruction configuration determines what the compiler emits, so it must be known at compile time.
//! A runtime branch cannot select different settings for a later instruction. MIR->vISA translation
//! must reject such a merge rather than silently use one arm's configuration.
//!
//! Runtime scalar restrictions are covered in [`super::runtime_if_scalar`].

use furiosa_opt_std::prelude::*;

pub use crate::vector_engine::A;

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// Runtime `if` selecting a VE operation: `if pick > 0 { AddFxp } else { SubFxp }`.
///
/// `pick` is loaded from a host-supplied scalar, so the branch cannot be resolved at compile time.
/// Its result supplies the operation kind in `vector_fxp(operation, 1)` after the branch;
/// the immediate operand remains the compile-time constant `1`.
///
/// Expected: `cannot merge FxpBinaryOp(SubFxp) with FxpBinaryOp(AddFxp)`
/// during MIR->vISA translation.
#[device(chip = 1)]
pub fn runtime_if_operation_setting(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A]>,
    pick: &HbmScalar<i32>,
) -> HbmTensor<i32, Chip, m![A]> {
    let pick = pick.to_spm(&mut device.tdma);
    let operation = if pick > 0 {
        FxpBinaryOp::AddFxp
    } else {
        FxpBinaryOp::SubFxp
    };

    let input = input.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);
    let output: DmTensor<i32, Chip, Cluster, Slice, m![A]> = device
        .main
        .begin(input.view())
        .fetch::<m![1], m![A]>()
        .fetch_cast::<i32>()
        .collect::<m![A / 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(operation, 1)
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit();
    output.to_hbm(&mut device.tdma)
}
