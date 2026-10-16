//! A loop whose iteration count arrives on a device tensor and whose body accumulates.
//!
//! The running sum is loop-carried. Each iteration reads it back into a register and adds the
//! next row onto that register, so no single operator both reads the carried value and produces it.

use furiosa_opt_std::prelude::*;

// A 4-PE device is 1 cluster x 256 slices, so `Cluster = m![1]` matches. `W / 4` x `W % 4` gives
// `Row` an 8-byte-aligned in-slice extent.
axes![W = 1024, Live = 8, One = 1];

type Chip = m![1];
type Cluster = m![1];
type Row = DmTensor<i32, Chip, Cluster, m![W / 4], m![W % 4]>;

/// Adds the leading `live` rows of `input` onto `init`, out of the `Live` the body is allocated for.
#[device(chip = 1, pe = 4)]
pub fn sum_live_rows(
    ctx: &mut Device,
    live: &HbmTensor<i32, Chip, m![One]>,
    init: &HbmTensor<i32, Chip, m![W]>,
    input: &HbmTensor<i32, Chip, m![Live, W]>,
) -> HbmTensor<i32, Chip, m![W]> {
    let rows: DmTensor<i32, Chip, Cluster, m![W / 4], m![Live, W % 4]> = input.to_dm(&mut ctx.tdma);
    let mut sum: Row = init.to_dm(&mut ctx.tdma);
    for row in 0..live.extent::<m![Live]>() {
        let sum_vrf: VrfTensor<i32, Chip, Cluster, m![W / 4], m![W % 4 # 8]> = ctx
            .sub
            .begin(sum.view())
            .fetch::<m![1], m![W % 4]>()
            .fetch_cast::<i32>()
            .collect::<m![1], m![W % 4 # 8]>()
            .to_vrf();
        sum = ctx
            .main
            .begin(rows.view().tile::<m![Live], 1, m![1 # 8, W % 4]>(row))
            .fetch::<m![1], m![W % 4]>()
            .fetch_cast::<i32>()
            .collect::<m![1], m![W % 4 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &sum_vrf)
            .vector_final()
            .commit_trim::<m![W % 4]>()
            .commit();
    }
    sum.to_hbm(&mut ctx.tdma)
}
