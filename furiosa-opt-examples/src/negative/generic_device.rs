//! Never-launched generic fns sharing the leaf name `gen_copy`: a `compile gen_copy` filter
//! matches both, and each match must be rejected per fn
//! (`generic_entrypoints_all_rejected` in npu-visa-test).

use furiosa_opt_std::prelude::*;

use crate::generic::B;

pub mod a {
    use super::*;

    #[device(chip = 1)]
    pub fn gen_copy<A: AxisName>(
        ctx: &mut Context,
        hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let dm: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut ctx.tdma);
        dm.to_hbm(&mut ctx.tdma)
    }
}

pub mod b {
    use super::*;

    #[device(chip = 1)]
    pub fn gen_copy<A: AxisName>(
        ctx: &mut Context,
        hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let dm: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut ctx.tdma);
        dm.to_hbm(&mut ctx.tdma)
    }
}
