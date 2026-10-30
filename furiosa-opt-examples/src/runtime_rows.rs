use furiosa_opt_std::prelude::*;

axes![Rows = 128, Red = 32, Out = 8, One = 1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

#[device(chip = 1)]
pub fn runtime_rows(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
    weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
    output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out]>(&mut ctx.tdma);
    let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Out], m![Red]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![Out, Red / 16], m![Red % 16]>()
        .collect::<m![Out, Red / 16], m![Red % 16]>()
        .to_trf();
    ctx.main
        .begin(input.view())
        .fetch::<m![Rows, Red / 16], m![Red % 16]>()
        .collect::<m![Rows, Red / 16], m![Red % 16]>()
        .contract_outer::<m![Rows], m![Red], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Rows]>()
        .contract_lane::<m![Rows], m![Out]>(LaneMode::Interleaved)
        .commit_trim::<m![Out]>()
        .commit_prefix::<m![Rows], _, _>(result.view_mut(), count.extent::<m![Rows]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
