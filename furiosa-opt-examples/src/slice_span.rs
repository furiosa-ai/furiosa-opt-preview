//! Tile DMAs whose slice span is a runtime power of two.
//!
//! A decode row spread over an aligned group of `2^depth` of a cluster's 256 slices needs its query
//! on every slice of the group and stores its result from the group's first slice. `query_span`
//! loads rows that way and `result_span` stores them that way, both from one runtime depth; the
//! static twins fix the span in the type.

use furiosa_opt_std::prelude::*;

axes![R = 256, E = 128, Depth = 8, One = 1];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Row `s >> depth` onto slice `s`: each group of `2^depth` slices receives one row.
#[device(chip = 1)]
pub fn query_span(
    ctx: &mut Device,
    depth: &HbmTensor<i32, Chip, m![One]>,
    rows: &HbmTensor<bf16, Chip, m![R, E]>,
) -> HbmTensor<bf16, Chip, m![R, E]> {
    let depth = depth.extent::<m![Depth]>();
    let spread = rows.view().to_dm_span::<Cluster, m![R], m![E]>(&mut ctx.tdma, depth);
    spread.to_hbm(&mut ctx.tdma)
}

/// Slice `s << depth` onto row `s`: the first slice of each group stores the group's row.
#[device(chip = 1)]
pub fn result_span(
    ctx: &mut Device,
    depth: &HbmTensor<i32, Chip, m![One]>,
    rows: &HbmTensor<bf16, Chip, m![R, E]>,
    out: &mut HbmTensor<bf16, Chip, m![R, E]>,
) {
    let depth = depth.extent::<m![Depth]>();
    let rows = rows.to_dm::<Cluster, m![R], m![E]>(&mut ctx.tdma);
    rows.view().to_hbm_view_span(&mut ctx.tdma, out.view_mut(), depth);
}

macro_rules! static_span {
    ($query:ident, $result:ident, $groups:ident, $lanes:ident) => {
        /// [`query_span`] with the span fixed in the type.
        #[device(chip = 1)]
        pub fn $query(
            ctx: &mut Device,
            rows: &HbmTensor<bf16, Chip, m![$groups, E]>,
        ) -> HbmTensor<bf16, Chip, m![$groups, $lanes, E]> {
            let spread = rows.to_dm::<Cluster, m![$groups, $lanes], m![E]>(&mut ctx.tdma);
            spread.to_hbm(&mut ctx.tdma)
        }

        /// [`result_span`] with the span fixed in the type.
        #[device(chip = 1)]
        pub fn $result(
            ctx: &mut Device,
            rows: &HbmTensor<bf16, Chip, m![$groups, $lanes, E]>,
            out: &mut HbmTensor<bf16, Chip, m![$groups, E]>,
        ) {
            let rows = rows.to_dm::<Cluster, m![$groups, $lanes], m![E]>(&mut ctx.tdma);
            rows.view()
                .slice_tile::<m![$lanes], 1, m![$groups, 1 # $lanes]>(0)
                .to_hbm_view(&mut ctx.tdma, out.view_mut());
        }
    };
}

axes![G1 = 128, L1 = 2, G4 = 16, L4 = 16, G6 = 4, L6 = 64, G8 = 1, L8 = 256];

static_span!(query_static_1, result_static_1, G1, L1);
static_span!(query_static_4, result_static_4, G4, L4);
static_span!(query_static_6, result_static_6, G6, L6);
static_span!(query_static_8, result_static_8, G8, L8);

/// The same spans over rows of several bursts a slice: a decode query row is 4 KiB a slice.
pub mod rows {
    use furiosa_opt_std::prelude::*;

    axes![R = 256, E = 2048, Depth = 8, One = 1];

    type Chip = m![1];
    type Cluster = m![1 # 2];

    #[device(chip = 1)]
    pub fn query_span(
        ctx: &mut Device,
        depth: &HbmTensor<i32, Chip, m![One]>,
        rows: &HbmTensor<bf16, Chip, m![R, E]>,
    ) -> HbmTensor<bf16, Chip, m![R, E]> {
        let depth = depth.extent::<m![Depth]>();
        rows.view()
            .to_dm_span::<Cluster, m![R], m![E]>(&mut ctx.tdma, depth)
            .to_hbm(&mut ctx.tdma)
    }

    #[device(chip = 1)]
    pub fn result_span(
        ctx: &mut Device,
        depth: &HbmTensor<i32, Chip, m![One]>,
        rows: &HbmTensor<bf16, Chip, m![R, E]>,
        out: &mut HbmTensor<bf16, Chip, m![R, E]>,
    ) {
        let depth = depth.extent::<m![Depth]>();
        let rows = rows.to_dm::<Cluster, m![R], m![E]>(&mut ctx.tdma);
        rows.view().to_hbm_view_span(&mut ctx.tdma, out.view_mut(), depth);
    }
}
