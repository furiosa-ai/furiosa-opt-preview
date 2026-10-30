use furiosa_opt_std::prelude::*;

axes![Group = 16, Lane = 8, Red = 32, Out = 2, One = 1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// Activation rows in the TRF (eight in the MAC lanes, row groups outermost in Time), weights
/// streamed by Main: the live row groups bound the contraction.
#[device(chip = 1)]
pub fn runtime_rows_trf(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], m![Group, Lane, Red]>,
    weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
    output: &mut HbmTensor<f32, m![1], m![Group, Out, Lane]>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Group, Lane, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Slice, m![Group, Out, Lane]>(&mut ctx.tdma);
    let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Lane], m![Group, Red]> = ctx
        .sub
        .begin(input.view())
        .fetch::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .collect::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .to_trf();
    ctx.main
        .begin(weight.view())
        .fetch::<m![Group, Red / 16], m![Out, Red % 16]>()
        .collect::<m![Group, Red / 16, Out], m![Red % 16]>()
        .contract_outer::<m![Group, Red / 16], m![Out, Red % 16], _, _, _>(&trf)
        .contract_packet::<m![Out]>()
        .contract_time::<m![Group]>()
        .contract_lane::<m![Group, Out], m![Lane]>(LaneMode::Interleaved)
        .commit_trim::<m![Lane]>()
        .commit_prefix::<m![Group], _, _>(result.view_mut(), count.extent::<m![Group]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
