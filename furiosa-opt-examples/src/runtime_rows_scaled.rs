use furiosa_opt_std::prelude::*;

axes![Bank = 16, Kp = 16, Group = 16, Lane = 8, Red = 16, Out = 16, One = 1];
type Cluster = m![1 # 2];
type Slice = m![Bank, Kp];
type Reduced = m![Bank, Group];

pub type Input = m![Bank, Kp, Group, Lane, Red];
pub type Weight = m![Bank, Kp, Out, Red];
pub type Scale = m![Bank, Kp, Group, Lane];
pub type Output = m![Bank, Group, Out / 4, Lane, Out % 4];

/// `runtime_rows_spread` with two outputs a weight packet, inside the row groups in time, and
/// each slice's partial sums scaled by its rows' own scales before the slices add them.
#[device(chip = 1)]
pub fn runtime_rows_scaled(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], Input>,
    weight: &HbmTensor<bf16, m![1], Weight>,
    scale: &HbmTensor<f32, m![1], Scale>,
    output: &mut HbmTensor<bf16, m![1], Output>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Group, Lane, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let scale = scale.to_dm::<Cluster, Slice, m![Group, Lane]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Reduced, m![Out / 4, Lane, Out % 4]>(&mut ctx.tdma);
    let scale: VrfTensor<f32, m![1], Cluster, Slice, m![Group, Lane]> = ctx
        .sub
        .begin(scale.view())
        .fetch::<m![Group], m![Lane]>()
        .collect::<m![Group], m![Lane]>()
        .to_vrf();
    let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Lane], m![Group, Red]> = ctx
        .sub
        .begin(input.view())
        .fetch::<m![Lane, Group], m![Red]>()
        .collect::<m![Lane, Group], m![Red]>()
        .to_trf();
    ctx.main
        .begin(weight.view())
        .fetch::<m![Out / 2], m![Out % 2, Red]>()
        .collect::<m![Out], m![Red]>()
        .contract_outer::<m![Out / 2, Group], m![Out % 2, Red], _, _, _>(&trf)
        .contract_packet::<m![Out % 2]>()
        .contract_time::<m![Out / 2, Group]>()
        .contract_lane::<m![Out / 2, Group, Out % 2], m![Lane]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Out / 2, Group, Out % 2, Lane / 4], m![Lane % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Out / 2, Group, Out % 2], m![Lane]>()
        .vector_inter_slice_reduce::<Reduced, m![Out]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Lane # 16]>()
        .transpose::<m![Out / 4, Lane], m![Out % 4 # 16]>()
        .commit_trim::<m![Out % 4]>()
        .commit_prefix::<m![Group], _, _>(result.view_mut(), count.extent::<m![Group]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}

/// [`runtime_rows_scaled`] with one output a weight packet, the row groups innermost in time.
#[device(chip = 1)]
pub fn runtime_rows_scaled_single(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], Input>,
    weight: &HbmTensor<bf16, m![1], Weight>,
    scale: &HbmTensor<f32, m![1], Scale>,
    output: &mut HbmTensor<bf16, m![1], Output>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Group, Lane, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let scale = scale.to_dm::<Cluster, Slice, m![Group, Lane]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Reduced, m![Out / 4, Lane, Out % 4]>(&mut ctx.tdma);
    let scale: VrfTensor<f32, m![1], Cluster, Slice, m![Group, Lane]> = ctx
        .sub
        .begin(scale.view())
        .fetch::<m![Group], m![Lane]>()
        .collect::<m![Group], m![Lane]>()
        .to_vrf();
    let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Lane], m![Group, Red]> = ctx
        .sub
        .begin(input.view())
        .fetch::<m![Lane, Group], m![Red]>()
        .collect::<m![Lane, Group], m![Red]>()
        .to_trf();
    ctx.main
        .begin(weight.view())
        .fetch::<m![Out], m![Red]>()
        .collect::<m![Out], m![Red]>()
        .contract_outer::<m![Out, Group], m![Red], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Out, Group]>()
        .contract_lane::<m![Out, Group], m![Lane]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Out, Group, Lane / 4], m![Lane % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Out, Group], m![Lane]>()
        .vector_inter_slice_reduce::<Reduced, m![Out]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Lane # 16]>()
        .transpose::<m![Out / 4, Lane], m![Out % 4 # 16]>()
        .commit_trim::<m![Out % 4]>()
        .commit_prefix::<m![Group], _, _>(result.view_mut(), count.extent::<m![Group]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
