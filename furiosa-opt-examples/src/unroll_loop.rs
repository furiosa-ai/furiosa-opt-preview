//! Kernels that exercise `#[unroll]` lowering.

use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32];

/// One row of the transpose: DMA row `b` of `input` into column `b` of `output`.
pub(crate) fn transpose_row(
    device: &mut Device,
    input: &HbmTensorView<'_, i8, m![1], m![A, B]>,
    output: &mut HbmTensor<i8, m![1], m![B, A]>,
    b: usize,
) {
    transpose_row_into(device, input, output, b, b)
}

fn transpose_row_into(
    device: &mut Device,
    input: &HbmTensorView<'_, i8, m![1], m![A, B]>,
    output: &mut HbmTensor<i8, m![1], m![B, A]>,
    from: usize,
    into: usize,
) {
    let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(from);
    let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(into);
    input_slice.to_hbm_view(&mut device.tdma, output_slice);
}

fn transpose_four_rows_in_helper(
    device: &mut Device,
    input: &HbmTensorView<'_, i8, m![1], m![A, B]>,
    output: &mut HbmTensor<i8, m![1], m![B, A]>,
) {
    #[unroll]
    for b in 0..4 {
        transpose_row(device, input, output, b);
    }
}

/// Transposes four rows while retaining the loop.
#[device(chip = 1)]
pub fn transpose_four_rows_rolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    for b in 0..4 {
        transpose_row(device, &input, &mut output, b);
    }
    output
}

/// Transposes four rows with an unrolled loop.
#[device(chip = 1)]
pub fn transpose_four_rows_unrolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    #[unroll]
    for b in 0..4 {
        transpose_row(device, &input, &mut output, b);
    }
    output
}

/// Transposes four rows in an unrolled loop inside a reachable helper.
#[device(chip = 1)]
pub fn transpose_four_rows_helper_unrolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    transpose_four_rows_in_helper(device, &input, &mut output);
    output
}

/// Runs a nested loop with both levels unrolled.
#[device(chip = 1)]
pub fn nested_both_unrolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    #[unroll]
    for i in 0..2 {
        #[unroll]
        for _j in 0..2 {
            transpose_row(device, &input, &mut output, i);
        }
    }
    output
}

axes![Cluster = 2, Slice = 256, Element = 16];

type Buffer = HbmTensor<bf16, m![1], m![Cluster, Slice, Element]>;
type Local = DmTensor<bf16, m![1], m![Cluster], m![Slice], m![Element]>;

fn copy(device: &mut Device, source: &Buffer, output: &mut Buffer) {
    let local: Local = source.to_dm(&mut device.tdma);
    local.view().to_hbm_view(&mut device.tdma, output.view_mut());
}

/// Copies `input` around unrolled loops over empty ranges, each of which would overwrite the
/// copy with `other` if it ran.
#[device(chip = 1)]
pub fn empty_ranges_unrolled(device: &mut Device, input: &Buffer, other: &Buffer, output: &mut Buffer) {
    #[unroll]
    for _ in 0..2 {
        copy(device, input, output);
        #[unroll]
        #[expect(clippy::reversed_empty_ranges, reason = "the empty range is the fixture")]
        for _ in 1..1 {
            copy(device, other, output);
        }
    }
    #[unroll]
    #[expect(clippy::reversed_empty_ranges, reason = "the empty range is the fixture")]
    for _ in 2..2 {
        copy(device, other, output);
    }
    #[unroll]
    #[expect(clippy::reversed_empty_ranges, reason = "the reversed range is the fixture")]
    for _ in 3..2 {
        copy(device, other, output);
    }
}

/// Stores `input`, which an unrolled loop over an empty range would have replaced with `output`.
#[device(chip = 1)]
pub fn empty_range_keeps_carried_tensor(device: &mut Device, input: &Buffer, output: &mut Buffer) {
    let mut value: Local = input.to_dm(&mut device.tdma);
    #[unroll]
    #[expect(clippy::reversed_empty_ranges, reason = "the empty range is the fixture")]
    for _ in 2..2 {
        value = output.to_dm(&mut device.tdma);
    }
    value.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
