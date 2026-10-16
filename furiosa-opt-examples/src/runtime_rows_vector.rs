use furiosa_opt_std::prelude::*;
axes![Rows = 128, H = 64, I = 2, One = 1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];
#[device(chip = 1)]
pub fn runtime_rows_add(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    lhs: &HbmTensor<f32, m![1], m![Rows, H]>,
    rhs: &HbmTensor<f32, m![1], m![Rows, H]>,
    output: &mut HbmTensor<f32, m![1], m![Rows, H]>,
) {
    let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
    let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view())
        .fetch::<m![Rows, H / 8, I], m![H % 8]>()
        .collect::<m![Rows, H / 8, I], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![Rows, H / 8]>()
        .vector_narrow_split::<m![Rows, H / 4], m![H % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<m![Rows, H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit_prefix::<m![Rows], _, _>(result.view_mut(), count.extent::<m![Rows]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
