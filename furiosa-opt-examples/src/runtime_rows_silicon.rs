//! Silicon twins of [`crate::runtime_rows`]: each runtime-row contraction beside the same kernel
//! committing every row statically, at the MRE size and at two projection-panel sizes.

/// The MRE's static, hoisted-count and two-contraction twins, all with its ABI.
pub mod mre {
    use furiosa_opt_std::prelude::*;

    use crate::runtime_rows::{One, Out, Red, Rows};
    type Cluster = m![1 # 2];
    type Slice = m![1 # 256];

    /// The MRE with a static full commit; `count` is unused so both kernels share one ABI.
    #[device(chip = 1)]
    pub fn runtime_rows_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
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
            .commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// The MRE with the live count read before any other statement.
    #[device(chip = 1)]
    pub fn runtime_rows_hoisted(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let live = count.extent::<m![Rows]>();
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
            .commit_prefix::<m![Rows], _, _>(result.view_mut(), live);
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// Two contractions of the MRE sharing one staged count, each into its own output.
    #[device(chip = 1)]
    pub fn runtime_rows_twice(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
        second: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let live = count.extent::<m![Rows]>();
        let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
        let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out]>(&mut ctx.tdma);
        let mut other = second.to_dm::<Cluster, Slice, m![Rows, Out]>(&mut ctx.tdma);
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
            .commit_prefix::<m![Rows], _, _>(result.view_mut(), live);
        ctx.main
            .begin(input.view())
            .fetch::<m![Rows, Red / 16], m![Red % 16]>()
            .collect::<m![Rows, Red / 16], m![Red % 16]>()
            .contract_outer::<m![Rows], m![Red], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rows]>()
            .contract_lane::<m![Rows], m![Out]>(LaneMode::Interleaved)
            .commit_trim::<m![Out]>()
            .commit_prefix::<m![Rows], _, _>(other.view_mut(), live);
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
        other.view().to_hbm_view(&mut ctx.tdma, second.view_mut());
    }

    /// [`runtime_rows_twice`] with static full commits.
    #[device(chip = 1)]
    pub fn runtime_rows_twice_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
        second: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
        let weight = weight.to_dm::<Cluster, Slice, m![Out, Red]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out]>(&mut ctx.tdma);
        let mut other = second.to_dm::<Cluster, Slice, m![Rows, Out]>(&mut ctx.tdma);
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
            .commit_view(result.view_mut());
        ctx.main
            .begin(input.view())
            .fetch::<m![Rows, Red / 16], m![Red % 16]>()
            .collect::<m![Rows, Red / 16], m![Red % 16]>()
            .contract_outer::<m![Rows], m![Red], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rows]>()
            .contract_lane::<m![Rows], m![Out]>(LaneMode::Interleaved)
            .commit_trim::<m![Out]>()
            .commit_view(other.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
        other.view().to_hbm_view(&mut ctx.tdma, second.view_mut());
    }
}

/// A K = 4096 panel into 128 columns over 16 slices, each reduction in one 64 KiB TRF. Every slice
/// holds the whole input at 8 KiB a row, so 32 rows fit where 128 rows would need 1 MiB.
pub mod qkv {
    use furiosa_opt_std::prelude::*;

    axes![Rows = 32, Red = 4096, Out = 128, One = 1];
    type Cluster = m![1 # 2];
    type Slice = m![Out / 8 # 256];

    #[device(chip = 1)]
    pub fn rows(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
        let weight = weight.to_dm::<Cluster, Slice, m![Out % 8, Red]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out % 8]>(&mut ctx.tdma);
        let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Out % 8], m![Red]> = ctx
            .sub
            .begin(weight.view())
            .fetch::<m![Out % 8, Red / 16], m![Red % 16]>()
            .collect::<m![Out % 8, Red / 16], m![Red % 16]>()
            .to_trf();
        ctx.main
            .begin(input.view())
            .fetch::<m![Rows, Red / 16], m![Red % 16]>()
            .collect::<m![Rows, Red / 16], m![Red % 16]>()
            .contract_outer::<m![Rows, Red / 32], m![Red % 32], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rows]>()
            .contract_lane::<m![Rows], m![Out % 8]>(LaneMode::Interleaved)
            .commit_trim::<m![Out % 8]>()
            .commit_prefix::<m![Rows], _, _>(result.view_mut(), count.extent::<m![Rows]>());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn rows_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
        let weight = weight.to_dm::<Cluster, Slice, m![Out % 8, Red]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out % 8]>(&mut ctx.tdma);
        let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Out % 8], m![Red]> = ctx
            .sub
            .begin(weight.view())
            .fetch::<m![Out % 8, Red / 16], m![Red % 16]>()
            .collect::<m![Out % 8, Red / 16], m![Red % 16]>()
            .to_trf();
        ctx.main
            .begin(input.view())
            .fetch::<m![Rows, Red / 16], m![Red % 16]>()
            .collect::<m![Rows, Red / 16], m![Red % 16]>()
            .contract_outer::<m![Rows, Red / 32], m![Red % 32], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rows]>()
            .contract_lane::<m![Rows], m![Out % 8]>(LaneMode::Interleaved)
            .commit_trim::<m![Out % 8]>()
            .commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

/// The same panel at the full 128-row capacity with K = 1024 (256 KiB of input a slice).
pub mod panel {
    use furiosa_opt_std::prelude::*;

    axes![Rows = 128, Red = 1024, Out = 128, One = 1];
    type Cluster = m![1 # 2];
    type Slice = m![Out / 8 # 256];

    #[device(chip = 1)]
    pub fn rows(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
        let weight = weight.to_dm::<Cluster, Slice, m![Out % 8, Red]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out % 8]>(&mut ctx.tdma);
        let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Out % 8], m![Red]> = ctx
            .sub
            .begin(weight.view())
            .fetch::<m![Out % 8, Red / 16], m![Red % 16]>()
            .collect::<m![Out % 8, Red / 16], m![Red % 16]>()
            .to_trf();
        ctx.main
            .begin(input.view())
            .fetch::<m![Rows, Red / 16], m![Red % 16]>()
            .collect::<m![Rows, Red / 16], m![Red % 16]>()
            .contract_outer::<m![Rows, Red / 32], m![Red % 32], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rows]>()
            .contract_lane::<m![Rows], m![Out % 8]>(LaneMode::Interleaved)
            .commit_trim::<m![Out % 8]>()
            .commit_prefix::<m![Rows], _, _>(result.view_mut(), count.extent::<m![Rows]>());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    pub fn rows_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
        let weight = weight.to_dm::<Cluster, Slice, m![Out % 8, Red]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out % 8]>(&mut ctx.tdma);
        let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Out % 8], m![Red]> = ctx
            .sub
            .begin(weight.view())
            .fetch::<m![Out % 8, Red / 16], m![Red % 16]>()
            .collect::<m![Out % 8, Red / 16], m![Red % 16]>()
            .to_trf();
        ctx.main
            .begin(input.view())
            .fetch::<m![Rows, Red / 16], m![Red % 16]>()
            .collect::<m![Rows, Red / 16], m![Red % 16]>()
            .contract_outer::<m![Rows, Red / 32], m![Red % 32], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rows]>()
            .contract_lane::<m![Rows], m![Out % 8]>(LaneMode::Interleaved)
            .commit_trim::<m![Out % 8]>()
            .commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// The panel with the live count read before any other statement.
    #[device(chip = 1)]
    pub fn rows_hoisted(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        input: &HbmTensor<bf16, m![1], m![Rows, Red]>,
        weight: &HbmTensor<bf16, m![1], m![Out, Red]>,
        output: &mut HbmTensor<f32, m![1], m![Rows, Out]>,
    ) {
        let live = count.extent::<m![Rows]>();
        let input = input.to_dm::<Cluster, Slice, m![Rows, Red]>(&mut ctx.tdma);
        let weight = weight.to_dm::<Cluster, Slice, m![Out % 8, Red]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, Out % 8]>(&mut ctx.tdma);
        let trf: TrfTensor<bf16, m![1], Cluster, Slice, m![Out % 8], m![Red]> = ctx
            .sub
            .begin(weight.view())
            .fetch::<m![Out % 8, Red / 16], m![Red % 16]>()
            .collect::<m![Out % 8, Red / 16], m![Red % 16]>()
            .to_trf();
        ctx.main
            .begin(input.view())
            .fetch::<m![Rows, Red / 16], m![Red % 16]>()
            .collect::<m![Rows, Red / 16], m![Red % 16]>()
            .contract_outer::<m![Rows, Red / 32], m![Red % 32], _, _, _>(&trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rows]>()
            .contract_lane::<m![Rows], m![Out % 8]>(LaneMode::Interleaved)
            .commit_trim::<m![Out % 8]>()
            .commit_prefix::<m![Rows], _, _>(result.view_mut(), live);
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}
