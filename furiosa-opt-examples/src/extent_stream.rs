//! One weight-streaming loop bounded three ways: unrolled over a compile-time count, kept as a
//! loop over a count read off a device tensor, and kept as such a loop over a body holding two
//! groups instead of one.
//!
//! The first two stage one weight group into the TRF per iteration and contract the activation
//! against it, so they show what the loop costs once the unroll is gone. The third shows that the
//! cost is the rolling rather than the run-time count: staging both of a pair before either
//! contracts buys the overlap back, and what the run-time count itself costs is the one fetch that
//! reads it.

use furiosa_opt_std::prelude::*;

axes![Tok = 16, Red = 64, Out = 8, Group = 20, Pairs = 10, One = 1];

type Chip = m![1];
type Cluster = m![Tok / 8 % 2];
type Slice = m![Tok % 8 # 256];
type Activation = DmTensor<bf16, Chip, Cluster, Slice, m![Red]>;
type Weight = DmTensor<bf16, Chip, Cluster, Slice, m![Group, Out, Red]>;
type Output = DmTensor<bf16, Chip, Cluster, Slice, m![Group, Out]>;
type WeightTrf = TrfTensor<bf16, Chip, Cluster, Slice, m![Out], m![Red]>;

/// Streams weight group `group` into the TRF.
fn stage_group(ctx: &mut Device, weight: &Weight, group: usize) -> WeightTrf {
    let weight_group = weight.view().tile::<m![Group], 1, m![1 # 20, Out, Red]>(group);
    ctx.sub
        .begin(weight_group)
        .fetch::<m![Out, Red / 16], m![Red % 16]>()
        .collect::<m![Out, Red / 16], m![Red % 16]>()
        .to_trf()
}

/// Contracts the activation against a staged weight group and writes group `group` of `output`.
fn contract_group(
    ctx: &mut Device,
    activation: &Activation,
    weight_trf: &WeightTrf,
    output: &mut Output,
    group: usize,
) {
    ctx.main
        .begin(activation.view())
        .fetch::<m![Red / 16], m![Red % 16]>()
        .collect::<m![Red / 16], m![Red % 16]>()
        .contract_outer::<m![Red / 32], m![Red % 32], _, _, _>(weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![Out]>(LaneMode::Interleaved)
        .cast::<bf16, m![Out # 16]>()
        .commit_trim::<m![Out]>()
        .commit_view(output.view_mut().tile::<m![Group], 1, m![1 #{!} 20, Out]>(group));
}

/// Every group, with the loop unrolled at compile time.
#[device(chip = 1)]
pub fn unrolled_groups(
    ctx: &mut Device,
    activation: &HbmTensor<bf16, Chip, m![Tok, Red]>,
    weight: &HbmTensor<bf16, Chip, m![Group, Out, Red]>,
) -> HbmTensor<bf16, Chip, m![Tok, Group, Out]> {
    let activation: Activation = activation.to_dm(&mut ctx.tdma);
    let weight: Weight = weight.to_dm(&mut ctx.tdma);
    let mut output = Output::new();

    #[unroll]
    for group in 0..Group::SIZE {
        let weight_trf = stage_group(ctx, &weight, group);
        contract_group(ctx, &activation, &weight_trf, &mut output, group);
    }

    let mut result = HbmTensor::<bf16, Chip, m![Tok, Group, Out]>::new();
    output.view().to_hbm_view(&mut ctx.tdma, result.view_mut());
    result
}

/// The leading `groups` groups, with the loop kept and its count read at run time; the groups
/// beyond the count are never written, so the result holds whatever their scratch held.
#[device(chip = 1)]
pub fn counted_groups(
    ctx: &mut Device,
    groups: &HbmTensor<i32, Chip, m![One]>,
    activation: &HbmTensor<bf16, Chip, m![Tok, Red]>,
    weight: &HbmTensor<bf16, Chip, m![Group, Out, Red]>,
) -> HbmTensor<bf16, Chip, m![Tok, Group, Out]> {
    let activation: Activation = activation.to_dm(&mut ctx.tdma);
    let weight: Weight = weight.to_dm(&mut ctx.tdma);
    let mut output = Output::new();

    for group in 0..groups.extent::<m![Group]>() {
        let weight_trf = stage_group(ctx, &weight, group);
        contract_group(ctx, &activation, &weight_trf, &mut output, group);
    }

    let mut result = HbmTensor::<bf16, Chip, m![Tok, Group, Out]>::new();
    output.view().to_hbm_view(&mut ctx.tdma, result.view_mut());
    result
}

/// The leading `pairs` pairs of groups, with the loop kept and its count read at run time.
///
/// Both groups of a pair are staged before either one contracts, which is what the loop body has to
/// spell out for the two to overlap: the scheduler pipelines within one body, never across the
/// iterations of a rolled loop, so a body holding a single group leaves each load exposed.
#[device(chip = 1)]
pub fn counted_pairs(
    ctx: &mut Device,
    pairs: &HbmTensor<i32, Chip, m![One]>,
    activation: &HbmTensor<bf16, Chip, m![Tok, Red]>,
    weight: &HbmTensor<bf16, Chip, m![Group, Out, Red]>,
) -> HbmTensor<bf16, Chip, m![Tok, Group, Out]> {
    let activation: Activation = activation.to_dm(&mut ctx.tdma);
    let weight: Weight = weight.to_dm(&mut ctx.tdma);
    let mut output = Output::new();

    for pair in 0..pairs.extent::<m![Pairs]>() {
        let first = pair * 2;
        let second = first + 1;
        let first_trf = stage_group(ctx, &weight, first);
        let second_trf = stage_group(ctx, &weight, second);
        contract_group(ctx, &activation, &first_trf, &mut output, first);
        contract_group(ctx, &activation, &second_trf, &mut output, second);
    }

    let mut result = HbmTensor::<bf16, Chip, m![Tok, Group, Out]>::new();
    output.view().to_hbm_view(&mut ctx.tdma, result.view_mut());
    result
}
