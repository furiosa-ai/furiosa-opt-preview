//! An array index must resolve to one value: a constant or a loop index, never a branch.

use furiosa_opt_std::prelude::*;

use crate::layer_pool::{Key, Width};

axes![A = 64];

type Chip = m![1];

#[device(chip = 1, pe = 1)]
pub fn branched(device: &mut Device, first: &HbmTensor<bf16, Chip, m![A]>, second: &HbmTensor<bf16, Chip, m![A]>) {
    let inputs = [first, second];
    for i in 0..2 {
        let _: DmTensor<bf16, Chip, m![1], m![1 # 64], m![A]> =
            inputs[if i == 0 { 0 } else { 1 }].to_dm(&mut device.tdma);
    }
}

/// Independent pool descriptors cannot be selected through a writable array alias.
#[device(chip = 1)]
pub fn pool_array(
    device: &mut Device,
    pools: &mut [Slice<crate::layer_pool::Block>; 4],
    indices: &HbmTensor<i32, Chip, m![Key]>,
    values: &HbmTensor<bf16, Chip, m![Key, Width]>,
) {
    let values: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = values.to_dm(&mut device.tdma);
    #[allow(
        clippy::needless_range_loop,
        reason = "the rejection fixture requires runtime indexing"
    )]
    for layer in 0..4 {
        pools[layer]
            .scatter::<m![1], m![1 # 2, 1 # 256, Width]>()
            .by_byte_offsets(indices.view())
            .from_dm_view(&mut device.tdma, values.view());
    }
}
