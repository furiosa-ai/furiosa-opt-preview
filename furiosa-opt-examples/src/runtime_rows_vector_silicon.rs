//! Silicon twins of [`crate::runtime_rows_vector`]: each runtime-row vector command beside the same
//! kernel committing every row statically, at the MRE size and at a bf16 size whose engine time
//! dominates the launch.

/// The MRE's static twin, with its ABI.
pub mod add {
    use furiosa_opt_std::prelude::*;

    use crate::runtime_rows_vector::{H, I, One, Rows};
    type Cluster = m![1 # 2];
    type Slice = m![1 # 256];

    /// The MRE with a static full commit; `count` is unused so both kernels share one ABI.
    #[device(chip = 1)]
    pub fn add_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
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
            .commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

/// A bf16 sum of two 128-row tiles whose vector command runs for tens of microseconds, as the
/// decode partial adds do, plus a kernel issuing two of them on one staged count.
pub mod wide {
    use furiosa_opt_std::prelude::*;

    axes![Rows = 128, H = 256, I = 2, One = 1];
    type Cluster = m![1 # 2];
    type Slice = m![1 # 256];
    // Expands to the sum up to its commit, so each kernel picks a static or a counted commit.
    macro_rules! sum {
        ($ctx:expr, $lhs:expr, $rhs:expr) => {
            $ctx.main
                .begin_interleaved::<I, _, _, _, _, _>($lhs, $rhs)
                .fetch::<m![Rows, H / 8, I], m![H % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows, H / 8, I], m![H % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<I, m![Rows, H / 8]>()
                .vector_narrow_split::<m![Rows, H / 4], m![H % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<m![Rows, H / 8], m![H % 8]>()
                .vector_final()
                .cast::<bf16, m![H % 8 # 16]>()
                .commit_trim::<m![H % 8]>()
        };
    }

    /// One sum committing the first `count` rows.
    #[device(chip = 1)]
    pub fn rows(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let live = count.extent::<m![Rows]>();
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_prefix::<m![Rows], _, _>(result.view_mut(), live);
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// The same sum committing every row; `count` is unused so both kernels share one ABI.
    #[device(chip = 1)]
    pub fn rows_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// `lhs + rhs` and then `(lhs + rhs) + rhs`, both on one staged count, into one output.
    #[device(chip = 1)]
    pub fn twice(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let live = count.extent::<m![Rows]>();
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let mut sum: DmTensor<bf16, m![1], Cluster, Slice, m![Rows, H]> = DmTensor::new();
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_prefix::<m![Rows], _, _>(sum.view_mut(), live);
        sum!(ctx, sum.view(), rhs.view()).commit_prefix::<m![Rows], _, _>(result.view_mut(), live);
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// The two sums committing every row.
    #[device(chip = 1)]
    pub fn twice_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let mut sum: DmTensor<bf16, m![1], Cluster, Slice, m![Rows, H]> = DmTensor::new();
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_view(sum.view_mut());
        sum!(ctx, sum.view(), rhs.view()).commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

/// A bf16 sum in the decode tokenwise layout: eight-row groups outermost with the operand pair
/// between a group and its rows, so one runtime row count shortens the command by whole groups.
pub mod grouped {
    use furiosa_opt_std::prelude::*;

    axes![Rows = 128, H = 32, I = 2, One = 1];
    type Cluster = m![1 # 2];
    type Slice = m![1 # 256];

    // Expands to the sum up to its commit, so each kernel picks a static or a counted commit.
    macro_rules! sum {
        ($ctx:expr, $lhs:expr, $rhs:expr) => {
            $ctx.main
                .begin_interleaved::<I, _, _, _, _, _>($lhs, $rhs)
                .fetch::<m![Rows / 8, I, Rows % 8, H / 8], m![H % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows / 8, I, Rows % 8, H / 8], m![H % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<I, m![Rows / 8, Rows % 8, H / 8]>()
                .vector_narrow_split::<m![Rows / 8, Rows % 8, H / 4], m![H % 4]>()
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Rows / 8, Rows % 8, H / 8], m![H % 8]>()
                .vector_final()
                .cast::<bf16, m![H % 8 # 16]>()
                .commit_trim::<m![H % 8]>()
        };
    }

    /// The sum over the groups holding the first `count` rows.
    #[device(chip = 1)]
    pub fn rows(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let groups = count.extent::<m![Rows]>().div_ceil(8);
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_prefix::<m![Rows / 8], _, _>(result.view_mut(), groups);
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// The same sum committing every row; `count` is unused so both kernels share one ABI.
    #[device(chip = 1)]
    pub fn rows_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

/// The grouped sum at the decode launch capacity: 512 rows spread as the tokenwise operators keep
/// them, sixteen hidden columns in each of 256 slices.
pub mod tokenwise {
    use furiosa_opt_std::prelude::*;

    axes![Rows = 512, H = 4096, I = 2, One = 1];
    type Cluster = m![1 # 2];
    type Slice = m![H / 16];

    // Expands to the sum up to its commit, so each kernel picks a static or a counted commit.
    macro_rules! sum {
        ($ctx:expr, $lhs:expr, $rhs:expr) => {
            $ctx.main
                .begin_interleaved::<I, _, _, _, _, _>($lhs, $rhs)
                .fetch::<m![Rows / 8, I, Rows % 8, H % 16 / 8], m![H % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows / 8, I, Rows % 8, H % 16 / 8], m![H % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<I, m![Rows / 8, Rows % 8, H % 16 / 8]>()
                .vector_narrow_split::<m![Rows / 8, Rows % 8, H % 16 / 4], m![H % 4]>()
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Rows / 8, Rows % 8, H % 16 / 8], m![H % 8]>()
                .vector_final()
                .cast::<bf16, m![H % 8 # 16]>()
                .commit_trim::<m![H % 8]>()
        };
    }

    /// The sum over the groups holding the first `count` rows.
    #[device(chip = 1)]
    pub fn rows(
        ctx: &mut Device,
        count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let groups = count.extent::<m![Rows]>().div_ceil(8);
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H % 16]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H % 16]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H % 16]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_prefix::<m![Rows / 8], _, _>(result.view_mut(), groups);
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }

    /// The same sum committing every row; `count` is unused so both kernels share one ABI.
    #[device(chip = 1)]
    pub fn rows_static(
        ctx: &mut Device,
        _count: &HbmTensor<i32, m![1], m![One]>,
        lhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        rhs: &HbmTensor<bf16, m![1], m![Rows, H]>,
        output: &mut HbmTensor<bf16, m![1], m![Rows, H]>,
    ) {
        let lhs = lhs.to_dm::<Cluster, Slice, m![Rows, H % 16]>(&mut ctx.tdma);
        let rhs = rhs.to_dm::<Cluster, Slice, m![Rows, H % 16]>(&mut ctx.tdma);
        let mut result = output.to_dm::<Cluster, Slice, m![Rows, H % 16]>(&mut ctx.tdma);
        sum!(ctx, lhs.view(), rhs.view()).commit_view(result.view_mut());
        result.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}
