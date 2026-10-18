//! Inter-slice reduces whose span is a runtime power of two.
//!
//! Every slice holds one key block of four heads × four queries; a wave reduces the scores' max and
//! sum and a `[4, 4, 128]` value tile across aligned groups of `2^depth` of the 256 slices, as an
//! attention row spread over that many slices does. The static twins fix the group size in the type.

use furiosa_opt_std::prelude::*;

axes![R = 256, H = 4, G = 4, D = 128, Depth = 8, One = 1];

type Chip = m![1];
type Cluster = m![1 # 2];

/// One reduce over a runtime span: the whole contract in one command.
#[device(chip = 1)]
pub fn max_span(
    ctx: &mut Device,
    depth: &HbmTensor<i32, Chip, m![One]>,
    scores: &HbmTensor<f32, Chip, m![R, H, G]>,
) -> HbmTensor<f32, Chip, m![R, H, G]> {
    let depth = depth.extent::<m![Depth]>();
    let scores = scores.to_dm::<Cluster, m![R], m![H, G]>(&mut ctx.tdma);
    let result: DmTensor<f32, Chip, Cluster, m![R], m![H, G]> = ctx
        .main
        .begin(scores.view())
        .fetch::<m![H / 2], m![H % 2, G]>()
        .collect::<m![H / 2], m![H % 2, G]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, depth)
        .vector_final()
        .commit_trim::<m![H % 2, G]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}

/// The three reduces of an attention wave over one runtime span, sharing its depth.
#[device(chip = 1)]
pub fn wave(
    ctx: &mut Device,
    depth: &HbmTensor<i32, Chip, m![One]>,
    scores: &HbmTensor<f32, Chip, m![R, H, G]>,
    values: &HbmTensor<f32, Chip, m![R, H, G, D]>,
) -> (
    HbmTensor<f32, Chip, m![R, H, G]>,
    HbmTensor<f32, Chip, m![R, H, G]>,
    HbmTensor<f32, Chip, m![R, H, G, D]>,
) {
    let scores = scores.to_dm::<Cluster, m![R], m![H, G]>(&mut ctx.tdma);
    let values = values.to_dm::<Cluster, m![R], m![H, G, D]>(&mut ctx.tdma);
    // Read after the loads, so the count's round trip runs under them.
    let depth = depth.extent::<m![Depth]>();
    let max: DmTensor<f32, Chip, Cluster, m![R], m![H, G]> = ctx
        .main
        .begin(scores.view())
        .fetch::<m![H / 2], m![H % 2, G]>()
        .collect::<m![H / 2], m![H % 2, G]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, depth)
        .vector_final()
        .commit_trim::<m![H % 2, G]>()
        .commit();
    let sum: DmTensor<f32, Chip, Cluster, m![R], m![H, G]> = ctx
        .main
        .begin(scores.view())
        .fetch::<m![H / 2], m![H % 2, G]>()
        .collect::<m![H / 2], m![H % 2, G]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, depth)
        .vector_final()
        .commit_trim::<m![H % 2, G]>()
        .commit();
    let pv: DmTensor<f32, Chip, Cluster, m![R], m![H, G, D]> = ctx
        .main
        .begin(values.view())
        .fetch::<m![H, G, D / 8], m![D % 8]>()
        .collect::<m![H, G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, depth)
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();
    (
        max.to_hbm(&mut ctx.tdma),
        sum.to_hbm(&mut ctx.tdma),
        pv.to_hbm(&mut ctx.tdma),
    )
}

macro_rules! static_wave {
    ($name:ident, $groups:ident, $lanes:ident) => {
        /// [`wave`] with the span fixed in the type: one result per group of `$lanes` slices.
        #[device(chip = 1)]
        pub fn $name(
            ctx: &mut Device,
            scores: &HbmTensor<f32, Chip, m![$groups, $lanes, H, G]>,
            values: &HbmTensor<f32, Chip, m![$groups, $lanes, H, G, D]>,
        ) -> (
            HbmTensor<f32, Chip, m![$groups, H, G]>,
            HbmTensor<f32, Chip, m![$groups, H, G]>,
            HbmTensor<f32, Chip, m![$groups, H, G, D]>,
        ) {
            let scores = scores.to_dm::<Cluster, m![$groups, $lanes], m![H, G]>(&mut ctx.tdma);
            let values = values.to_dm::<Cluster, m![$groups, $lanes], m![H, G, D]>(&mut ctx.tdma);
            let max: DmTensor<f32, Chip, Cluster, m![$groups, 1 # $lanes], m![H, G]> = ctx
                .main
                .begin(scores.view())
                .fetch::<m![H / 2], m![H % 2, G]>()
                .collect::<m![H / 2], m![H % 2, G]>()
                .vector_init()
                .vector_inter_slice_reduce::<m![$groups, 1 # $lanes], m![H / 2]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![H % 2, G]>()
                .commit();
            let sum: DmTensor<f32, Chip, Cluster, m![$groups, 1 # $lanes], m![H, G]> = ctx
                .main
                .begin(scores.view())
                .fetch::<m![H / 2], m![H % 2, G]>()
                .collect::<m![H / 2], m![H % 2, G]>()
                .vector_init()
                .vector_inter_slice_reduce::<m![$groups, 1 # $lanes], m![H / 2]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![H % 2, G]>()
                .commit();
            let pv: DmTensor<f32, Chip, Cluster, m![$groups, 1 # $lanes], m![H, G, D]> = ctx
                .main
                .begin(values.view())
                .fetch::<m![H, G, D / 8], m![D % 8]>()
                .collect::<m![H, G, D / 8], m![D % 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<m![$groups, 1 # $lanes], m![H, G, D / 8]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![D % 8]>()
                .commit();
            (max.to_hbm(&mut ctx.tdma), sum.to_hbm(&mut ctx.tdma), pv.to_hbm(&mut ctx.tdma))
        }
    };
}

// Each group and its lanes are separate axes: a reduce over one split of a single axis is not lowered
// to its ratio.
axes![G5 = 8, L5 = 32, G6 = 4, L6 = 64, G7 = 2, L7 = 128, G8 = 1, L8 = 256];

static_wave!(wave_static_5, G5, L5);
static_wave!(wave_static_6, G6, L6);
static_wave!(wave_static_7, G7, L7);
static_wave!(wave_static_8, G8, L8);

/// A runtime-span max loaded into the register file by Main, then subtracted at each slice: every
/// slice reads its own group's max, as a decode row's peak feeds its scores' exponent.
#[device(chip = 1)]
pub fn max_span_to_vrf(
    ctx: &mut Device,
    depth: &HbmTensor<i32, Chip, m![One]>,
    peaks: &HbmTensor<f32, Chip, m![R, 1 # 8]>,
    scores: &HbmTensor<f32, Chip, m![R, H, G]>,
) -> HbmTensor<f32, Chip, m![R, H, G]> {
    let depth = depth.extent::<m![Depth]>();
    let peaks = peaks.to_dm::<Cluster, m![R], m![1 # 8]>(&mut ctx.tdma);
    let scores = scores.to_dm::<Cluster, m![R], m![H, G]>(&mut ctx.tdma);
    let peak: VrfTensor<f32, Chip, Cluster, m![R], m![1 # 8]> = ctx
        .main
        .begin(peaks.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, depth)
        .vector_final()
        .to_vrf(&mut ctx.sub);
    let result: DmTensor<f32, Chip, Cluster, m![R], m![H, G]> = ctx
        .main
        .begin(scores.view())
        .fetch::<m![H / 2], m![H % 2, G]>()
        .collect::<m![H / 2], m![H % 2, G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H], m![G]>()
        .vector_fp_binary(FpBinaryOp::SubF, &peak)
        .vector_widen_concat::<m![H / 2], m![H % 2, G]>()
        .vector_final()
        .commit_trim::<m![H % 2, G]>()
        .commit();
    result.to_hbm(&mut ctx.tdma)
}
