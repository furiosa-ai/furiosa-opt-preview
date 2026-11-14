//! FMA operand-slot examples: a `TagMode::Comparison` tag drives a different ternary operand per
//! branch, so each kernel below pins which slot carries which `(mul, add)` pair.
//!
//! The open question these answer is the RF branch's addend. An RF operand takes its multiplicand
//! from the register file and has no add register of its own, so the lowering has to say where its
//! addend comes from: the scalar the branch declared, or whatever add register 0 already holds.
//! [`ve_fma_operand_reg0_and_rf`] gives the two a different value, so the output says which.
use super::*;
// Bit requirements read best unqualified inside a pattern literal.
use furiosa_opt_std::prelude::BitReq::{Ignore, One, Zero};

// Prefixed rather than reusing the parent module's names, which are declared at other sizes.
axes![FP = 256, FA = 4, FT = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Tag bit 0: the element is zero.
fn zero() -> TagGuard {
    TagGuard::matches([One, Ignore, Ignore, Ignore])
}

/// Tag bits 1 set and 3 clear, with bit 0 clear: the element is positive and at most 100.
fn positive() -> TagGuard {
    TagGuard::matches([Zero, One, Ignore, Zero])
}

/// Tag bits 1 and 3 both set: the element is above 100.
fn big() -> TagGuard {
    TagGuard::matches([Zero, One, Ignore, One])
}

/// Tag bit 2 with bits 0 and 1 clear: the element is negative.
fn negative() -> TagGuard {
    TagGuard::matches([Zero, Zero, One, Ignore])
}

/// Stages the per-element scale the RF branches read, in the sub context.
fn scale_to_vrf(
    device: &mut Device,
    scale: &HbmTensor<f32, Chip, m![FP, FT]>,
) -> VrfTensor<f32, Chip, Cluster, m![FP], m![FT]> {
    let scale_dm = scale.to_dm::<Cluster, m![FP], m![FT]>(&mut device.tdma);
    device
        .sub
        .begin(scale_dm.view())
        .fetch::<m![1], m![FT]>()
        .collect::<m![1], m![FT]>()
        .to_vrf()
}

/// Shapes the tag stream every kernel here starts from.
macro_rules! tagged {
    ($device:expr, $input_dm:expr) => {
        $device
            .main
            .begin($input_dm.view())
            .fetch::<m![FA], m![FT]>()
            .collect::<m![FA], m![FT]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Equal(0.0),
                Cmp::Greater(0.0),
                Cmp::Less(0.0),
                Cmp::Greater(100.0),
            ]))
            .vector_narrow_split::<m![FA, FT / 4], m![FT % 4]>()
    };
}

/// One immediate pair in slot `Reg0`: positives get `x * 2 + 10`, everything else passes through.
#[device(chip = 1)]
pub fn ve_fma_operand_reg0(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![FA], m![FT]>()
        .collect::<m![FA], m![FT]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Equal(0.0),
            Cmp::Greater(0.0),
            Cmp::Less(0.0),
            Cmp::Greater(100.0),
        ]))
        .vector_narrow_split::<m![FA, FT / 4], m![FT % 4]>()
        .vector_fp_ternary(FpTernaryOp::FmaF, Branched::imm(positive(), (2.0f32, 10.0f32)))
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// An immediate pair in `Reg0` alongside an RF branch, with different addends on purpose.
///
/// `Reg0` guards the zero elements with addend `7.0`; the RF branch guards the positives and
/// declares addend `1000.0`. A positive element therefore reports which addend the RF branch reads.
#[device(chip = 1)]
pub fn ve_fma_operand_reg0_and_rf(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
    scale: &HbmTensor<f32, Chip, m![FP, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);
    let scale_vrf = scale_to_vrf(device, scale);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = tagged!(device, input_dm)
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(zero(), (0.0f32, 7.0f32)).rf(positive(), (&scale_vrf, 7.0f32)),
        )
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// Two immediate pairs: `Reg0` guards the positives, `Reg1` the negatives.
#[device(chip = 1)]
pub fn ve_fma_operand_two_regs(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = tagged!(device, input_dm)
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(positive(), (2.0f32, 10.0f32)).imm(negative(), (-3.0f32, 100.0f32)),
        )
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// Every immediate pair: `Reg0` the positives, `Reg1` the negatives, `Reg2` the zeros.
#[device(chip = 1)]
pub fn ve_fma_operand_three_regs(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = tagged!(device, input_dm)
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(positive(), (2.0f32, 10.0f32))
                .imm(negative(), (-3.0f32, 100.0f32))
                .imm(zero(), (0.0f32, 7.0f32)),
        )
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// The RF branch alone, so its declared addend is the only addend in the pass.
#[device(chip = 1)]
pub fn ve_fma_operand_rf(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
    scale: &HbmTensor<f32, Chip, m![FP, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);
    let scale_vrf = scale_to_vrf(device, scale);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = tagged!(device, input_dm)
        .vector_fp_ternary(FpTernaryOp::FmaF, Branched::rf(positive(), (&scale_vrf, 1000.0f32)))
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// Two immediate pairs plus an RF branch, which is the most slots an RF pass can hold if the RF
/// branch takes one of the three add registers.
#[device(chip = 1)]
pub fn ve_fma_operand_two_regs_and_rf(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
    scale: &HbmTensor<f32, Chip, m![FP, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);
    let scale_vrf = scale_to_vrf(device, scale);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = tagged!(device, input_dm)
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(zero(), (0.0f32, 7.0f32))
                .imm(negative(), (-3.0f32, 100.0f32))
                .rf(positive(), (&scale_vrf, 1000.0f32)),
        )
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// Every immediate pair plus an RF branch: four operand slots against three add registers.
#[device(chip = 1)]
pub fn ve_fma_operand_three_regs_and_rf(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
    scale: &HbmTensor<f32, Chip, m![FP, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);
    let scale_vrf = scale_to_vrf(device, scale);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = tagged!(device, input_dm)
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(zero(), (0.0f32, 7.0f32))
                .imm(negative(), (-3.0f32, 100.0f32))
                .imm(big(), (5.0f32, 50.0f32))
                .rf(positive(), (&scale_vrf, 7.0f32)),
        )
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// `Reg0` and the RF branch guard the same elements, so the output names the winner. Slot order is
/// `Reg0`, `Reg1`, `Reg2`, then RF, which puts the immediate first.
#[device(chip = 1)]
pub fn ve_fma_operand_reg0_over_rf(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![FP, FA, FT]>,
    scale: &HbmTensor<f32, Chip, m![FP, FT]>,
) -> HbmTensor<f32, Chip, m![FP, FA, FT]> {
    let input_dm = input.to_dm::<Cluster, m![FP], m![FA, FT]>(&mut device.tdma);
    let scale_vrf = scale_to_vrf(device, scale);

    let result: DmTensor<f32, Chip, Cluster, m![FP], m![FA, FT]> = tagged!(device, input_dm)
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(positive(), (2.0f32, 10.0f32)).rf(positive(), (&scale_vrf, 10.0f32)),
        )
        .vector_widen_concat::<m![FA], m![FT]>()
        .vector_final()
        .commit_trim::<m![FT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}
