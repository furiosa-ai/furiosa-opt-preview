use furiosa_opt_std::prelude::*;

axes![Bank = 16, Kp = 16, Group = 16, Lane = 8, Red = 32, Out = 16, One = 1];
type Cluster = m![1 # 2];
type Slice = m![Bank, Kp];
type Reduced = m![Bank, Group];

pub type Input = m![Bank, Kp, Group, Lane, Red];
pub type Weight = m![Bank, Kp, Out, Red];
pub type Output = m![Bank, Group, Out / 4, Lane, Out % 4];

/// The Q/K/V projection shape in small: TRF rows streamed innermost so each weight packet serves
/// every row group, K split over sixteen slices whose reduce spreads the row groups over them.
#[device(chip = 1)]
pub fn runtime_rows_spread(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], Input>,
    weight: &HbmTensor<bf16, m![1], Weight>,
    output: &mut HbmTensor<bf16, m![1], Output>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Group, Lane, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Reduced, m![Out / 4, Lane, Out % 4]>(&mut ctx.tdma);
    let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Lane], m![Group, Red]> = ctx
        .sub
        .begin(input.view())
        .fetch::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .collect::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .to_trf();
    ctx.main
        .begin(weight.view())
        .fetch::<m![Out], m![Red]>()
        .collect::<m![Out, Red / 16], m![Red % 16]>()
        .contract_outer::<m![Out, Group], m![Red], _, _, _>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Out, Group]>()
        .contract_lane::<m![Out, Group], m![Lane]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Reduced, m![Out]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Lane # 16]>()
        .transpose::<m![Out / 4, Lane], m![Out % 4 # 16]>()
        .commit_trim::<m![Out % 4]>()
        .commit_prefix::<m![Group], _, _>(result.view_mut(), count.extent::<m![Group]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
