use furiosa_opt_std::prelude::*;

axes![Rows = 128, Hidden = 4096, X = 256, One = 1];
type Cluster = m![1 # 2];

/// A norm's variance pass over the live rows: each row's square sum reduces within a slice, then
/// across the 256 slices of a cluster, and lands broadcast on every slice.
#[device(chip = 1)]
pub fn runtime_rows_norm(
    ctx: &mut Device,
    count: &HbmTensor<i32, m![1], m![One]>,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
    output: &mut HbmTensor<f32, m![1], m![X, Rows, 1 # 8]>,
) {
    let input = input.to_dm::<Cluster, m![Hidden / 16], m![Rows, Hidden % 16]>(&mut ctx.tdma);
    let mut result = output.to_dm::<Cluster, m![X], m![Rows, 1 # 8]>(&mut ctx.tdma);
    ctx.main
        .begin(input.view())
        .fetch::<m![Rows, Hidden % 16 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows, Hidden % 16 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows, Hidden % 16 / 4], m![Hidden % 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::MulF(FpMulAlu::Mul0), BinaryArgMode::Mode00, 0.0f32)
        .vector_intra_slice_reduce::<Hidden, m![Rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![X], m![Rows]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_prefix::<m![Rows], _, _>(result.view_mut(), count.extent::<m![Rows]>());
    result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
}
