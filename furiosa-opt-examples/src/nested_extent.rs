//! Two loops whose iteration counts arrive on device tensors, one nested in the other.
//!
//! Each level declares its own envelope: `Outer` and `Inner` are the iterations its body is
//! allocated for, and its count tensor holds how many of them run.

use furiosa_opt_std::prelude::*;

axes![Wide = 8192, Outer = 4, Inner = 4, One = 1];

type Chip = m![1];

/// Copies tile `outer * Inner + inner` of 512 elements into `output` for the leading `outers` x
/// `inners` pairs, out of the `Outer` x `Inner` the body is allocated for; every other tile of
/// `output` is left as the caller filled it.
#[device(chip = 1)]
pub fn copy_live_grid(
    context: &mut Device,
    outers: &HbmTensor<i32, Chip, m![One]>,
    inners: &HbmTensor<i32, Chip, m![One]>,
    input: &HbmTensor<bf16, Chip, m![Wide]>,
    output: &mut HbmTensor<bf16, Chip, m![Wide]>,
) {
    for outer in 0..outers.extent::<m![Outer]>() {
        for inner in 0..inners.extent::<m![Inner]>() {
            let tile = outer * Inner::SIZE + inner;
            let source = input.view().tile::<m![Wide], 512, m![Wide = 512 # 8192]>(tile * 512);
            let destination = output
                .view_mut()
                .tile::<m![Wide], 512, m![Wide = 512 #{!} 8192]>(tile * 512);
            source.to_hbm_view(&mut context.tdma, destination);
        }
    }
}
