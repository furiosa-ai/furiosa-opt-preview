//! A loop whose iteration count arrives on a device tensor.
//!
//! The count is read where the loop consumes it, so the call site declares both halves the machine
//! needs: `Live` is the iterations the body is allocated for, and the tensor holds how many run.

use furiosa_opt_std::prelude::*;

axes![Wide = 8192, Live = 16, One = 1, Rows = 4];

// A scalar row keeps two i32 cells to satisfy the 8-byte SRAM access width.
axes![Access = 2];

type Chip = m![1];

/// Copies the leading `live` tiles of 512 elements, out of the `Live` the body is allocated for;
/// the tiles beyond the count stay zero.
#[device(chip = 1)]
pub fn copy_live_tiles(
    context: &mut Device,
    live: &HbmTensor<i32, Chip, m![One]>,
    input: &HbmTensor<bf16, Chip, m![Wide]>,
) -> HbmTensor<bf16, Chip, m![Wide]> {
    let mut output = HbmTensor::<bf16, Chip, m![Wide]>::new();
    for tile in 0..live.extent::<m![Live]>() {
        let source = input.view().tile::<m![Wide], 512, m![Wide = 512 # 8192]>(tile * 512);
        let destination = output
            .view_mut()
            .tile::<m![Wide], 512, m![Wide = 512 #{!} 8192]>(tile * 512);
        source.to_hbm_view(&mut context.tdma, destination);
    }
    output
}

/// Copies leading tiles for each row, with that row's count read at the index the outer loop gives.
///
/// A batch of unequal work wants this: the inner loop runs only as many iterations as one row
/// needs, so the cost is the sum over the rows rather than the rows times the largest count.
/// `extent` reads a tensor's first element and takes an owned tensor, so one row's count is
/// materialised on its own before the loop that consumes it reads it.
#[device(chip = 1)]
pub fn copy_live_tiles_per_row(
    context: &mut Device,
    rows: &HbmTensor<i32, Chip, m![One]>,
    live: &HbmTensor<i32, Chip, m![Rows, Access]>,
    input: &HbmTensor<bf16, Chip, m![Wide]>,
    output: &mut HbmTensor<bf16, Chip, m![Wide]>,
) {
    for row in 0..rows.extent::<m![Rows]>() {
        let held: DmTensor<i32, Chip, m![1 # 2], m![1 # 256], m![Access]> = live
            .view()
            .tile::<m![Rows], 1, m![1 # 4, Access]>(row)
            .to_dm(&mut context.tdma);
        let held: HbmTensor<i32, Chip, m![Access]> = held.to_hbm(&mut context.tdma);
        for tile in 0..held.extent::<m![Live]>() {
            let source = input.view().tile::<m![Wide], 512, m![Wide = 512 # 8192]>(tile * 512);
            let destination = output
                .view_mut()
                .tile::<m![Wide], 512, m![Wide = 512 #{!} 8192]>(tile * 512);
            source.to_hbm_view(&mut context.tdma, destination);
        }
    }
}

axes![Layers = 3];

/// Copies tiles after the first, leaving every tile outside the runtime count untouched.
#[device(chip = 1)]
pub fn copy_remaining(
    device: &mut Device,
    count: &HbmTensor<i32, Chip, m![One]>,
    input: &HbmTensor<bf16, Chip, m![Wide]>,
    output: &mut HbmTensor<bf16, Chip, m![Wide]>,
) {
    for tile in 1..count.extent::<m![Live]>() {
        input
            .view()
            .tile::<m![Wide], 512, m![Wide = 512 # 8192]>(tile * 512)
            .to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![Wide], 512, m![Wide = 512 #{!} 8192]>(tile * 512),
            );
    }
}

/// Arithmetic spelling of the same bound, for image comparison.
#[device(chip = 1)]
pub fn copy_remaining_control(
    device: &mut Device,
    count: &HbmTensor<i32, Chip, m![One]>,
    input: &HbmTensor<bf16, Chip, m![Wide]>,
    output: &mut HbmTensor<bf16, Chip, m![Wide]>,
) {
    for tile in 1..count.extent::<m![Live]>() {
        input
            .view()
            .tile::<m![Wide], 512, m![Wide = 512 # 8192]>(tile * 512)
            .to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![Wide], 512, m![Wide = 512 #{!} 8192]>(tile * 512),
            );
    }
}

/// Copies rows `0` and `stride` of `input`, where `stride` is a count read from a device tensor.
#[device(chip = 1)]
pub fn copy_strided_rows(
    context: &mut Device,
    stride: &HbmTensor<i32, Chip, m![One]>,
    input: &HbmTensor<bf16, Chip, m![Layers, Wide]>,
    output: &mut HbmTensor<bf16, Chip, m![Layers, Wide]>,
) {
    let stride = stride.extent::<m![2]>();
    for row in 0..2 {
        let source = input
            .view()
            .tile::<m![Layers], 1, m![Layers = 1 # Layers, Wide]>(row * stride);
        let destination = output
            .view_mut()
            .tile::<m![Layers], 1, m![Layers = 1 #{!} Layers, Wide]>(row);
        source.to_hbm_view(&mut context.tdma, destination);
    }
}
