use furiosa_opt_std::prelude::*;

axes![Bank = 64, Group = 16, Lane = 8, Kp = 4, Red = 32, Out = 16, One = 1];
type Cluster = m![1 # 2];
type Slice = m![Bank, Kp];
type Reduced = m![Bank, Out / 4];

pub type Input = m![Bank, Kp, Group, Lane, Red];
pub type Weight = m![Bank, Kp, Out, Red];
pub type Output = m![Bank, Out / 4, Group, Lane, Out % 4];

/// The MLP gate shape in small: TRF rows (eight lanes, row groups outermost in Time), K split over
/// four slices and reduced into one output quarter per slice, the lanes transposed into rows.
#[device(chip = 1)]
pub fn runtime_rows_reduce(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], Input>,
    weight: &HbmTensor<bf16, m![1], Weight>,
    output: &mut HbmTensor<bf16, m![1], Output>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Group, Lane, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Reduced, m![Group, Lane, Out % 4]>(&mut ctx.tdma);
    let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Lane], m![Group, Red]> = ctx
        .sub
        .begin(input.view())
        .fetch::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .collect::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .to_trf();
    ctx.main
        .begin(weight.view())
        .fetch::<m![Group, Out / 2, Red / 16], m![Out % 2, Red % 16]>()
        .collect::<m![Group, Out / 2, Red / 16, Out % 2], m![Red % 16]>()
        .contract_outer::<m![Group, Out / 2, Red / 16], m![Out % 2, Red % 16], _, _, _>(&trf)
        .contract_packet::<m![Out % 2]>()
        .contract_time::<m![Group, Out / 2]>()
        .contract_lane::<m![Group, Out / 2, Out % 2], m![Lane]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Reduced, m![Group, Out % 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Lane # 16]>()
        .transpose::<m![Group, Lane], m![Out % 4 # 16]>()
        .commit_trim::<m![Out % 4]>()
        .commit_prefix::<m![Group], _, _>(result.view_mut(), count.extent::<m![Group]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}

/// [`runtime_rows_reduce`] committing every row group.
#[device(chip = 1)]
pub fn runtime_rows_reduce_static(
    ctx: &mut Device,
    _count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], Input>,
    weight: &HbmTensor<bf16, m![1], Weight>,
    output: &mut HbmTensor<bf16, m![1], Output>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Group, Lane, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Reduced, m![Group, Lane, Out % 4]>(&mut ctx.tdma);
    let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Lane], m![Group, Red]> = ctx
        .sub
        .begin(input.view())
        .fetch::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .collect::<m![Lane, Group, Red / 16], m![Red % 16]>()
        .to_trf();
    ctx.main
        .begin(weight.view())
        .fetch::<m![Group, Out / 2, Red / 16], m![Out % 2, Red % 16]>()
        .collect::<m![Group, Out / 2, Red / 16, Out % 2], m![Red % 16]>()
        .contract_outer::<m![Group, Out / 2, Red / 16], m![Out % 2, Red % 16], _, _, _>(&trf)
        .contract_packet::<m![Out % 2]>()
        .contract_time::<m![Group, Out / 2]>()
        .contract_lane::<m![Group, Out / 2, Out % 2], m![Lane]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Reduced, m![Group, Out % 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Lane # 16]>()
        .transpose::<m![Group, Lane], m![Out % 4 # 16]>()
        .commit_trim::<m![Out % 4]>()
        .commit_view(result.view_mut());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
