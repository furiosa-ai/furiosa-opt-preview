use furiosa_opt_std::prelude::*;

axes![Bank = 64, Rows = 256, Kp = 4, Red = 32, Out = 32, One = 1];
type Cluster = m![1 # 2];
type Slice = m![Bank, Kp];
type Reduced = m![Bank, Out / 8];
type Wave = m![Rows = 128];

pub type Input = m![Bank, Kp, Rows, Red];
pub type Weight = m![Bank, Kp, Out, Red];
pub type Output = m![Bank, Out / 8, Rows, Out % 8];

/// Two 128-row waves of one owner, each a counted contraction over its live 32-row quarters
/// committed into the wave's write tile; the two commands are alike but for their tile.
#[device(chip = 1)]
pub fn runtime_rows_tiles(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], Input>,
    weight: &HbmTensor<bf16, m![1], Weight>,
    output: &mut HbmTensor<bf16, m![1], Output>,
) {
    let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
    let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Reduced, m![Rows, Out % 8]>(&mut ctx.tdma);
    let rows = count.extent::<m![Rows]>();
    #[unroll]
    for wave in 0..2 {
        let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Rows % 8], m![{ Wave } / 32, Rows % 32 / 8, Red]> = ctx
            .sub
            .begin(
                input
                    .view()
                    .tile::<m![Rows], 128, m![Rows = 128 # Rows, Red]>(wave * 128)
                    .unpad::<m![{ Wave }, Red]>(),
            )
            .fetch::<m![Rows % 8, { Wave } / 32, Rows % 32 / 8, Red / 16], m![Red % 16]>()
            .collect::<m![Rows % 8, { Wave } / 32, Rows % 32 / 8, Red / 16], m![Red % 16]>()
            .to_trf();
        ctx.main
            .begin(weight.view())
            .fetch::<m![{ Wave } / 32, Out, Red / 16], m![Red % 16]>()
            .collect::<m![{ Wave } / 32, Out, Red / 16], m![Red % 16]>()
            .contract_outer::<m![{ Wave } / 32, Out, Red / 16, Rows % 32 / 8], m![Red % 16], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Wave } / 32, Out, Rows % 32 / 8]>()
            .contract_lane::<m![{ Wave } / 32, Out, Rows % 32 / 8], m![Rows % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced, m![{ Wave } / 32, Out % 8, Rows % 32 / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![Rows % 8 # 16]>()
            .transpose::<m![{ Wave } / 32, Out % 8 / 4, Rows % 32], m![Out % 4 # 16]>()
            .commit_trim::<m![Out % 4]>()
            .commit_prefix::<m![{ Wave } / 32], _, _>(
                result
                    .view_mut()
                    .tile::<m![Rows], 128, m![Rows = 128 #{!} Rows, Out % 8]>(wave * 128),
                (rows.max(128 * wave) + 31 - 128 * wave) / 32,
            );
    }
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
