//! Commits that write part of the slices: a Slice tile keeps the values of the slices outside it, a
//! new tensor leaves its `Bottom` slices unwritten, and a stream may hold more valid slices than its
//! destination.

use furiosa_opt_std::prelude::*;

axes![A = 8, B = 8, O = 8, E = 4];

/// Adds one to the first half of every `B` row and leaves the second half as it was.
#[device(chip = 1, pe = 1)]
pub fn add_one_in_a_slice_tile(
    device: &mut Device,
    input: &HbmTensor<i32, m![1], m![A, B, O]>,
    previous: &HbmTensor<i32, m![1], m![A, B, O]>,
) -> HbmTensor<i32, m![1], m![A, B, O]> {
    let source: DmTensor<i32, m![1], m![1], m![A, B], m![O]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, m![1], m![1], m![A, B], m![O]> = previous.to_dm(&mut device.tdma);
    let source = source.view().slice_tile::<m![B / 4], 1, m![A, B % 4 # 8]>(0);
    let destination = output.view_mut().slice_tile::<m![B / 4], 1, m![A, B % 4 #{!} 8]>(0);
    device
        .main
        .begin(source)
        .fetch::<m![1], m![O]>()
        .collect::<m![1], m![O]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![O]>()
        .commit_view(destination);
    output.to_hbm(&mut device.tdma)
}

/// Commits to a new tensor whose `B >= 4` slices are `Bottom` padding.
#[device(chip = 1, pe = 1)]
pub fn copy_into_bottom_padded_slices(
    device: &mut Device,
    input: &HbmTensor<i32, m![1], m![A, B = 4, O]>,
) -> HbmTensor<i32, m![1], m![A, B = 4, O]> {
    let source: DmTensor<i32, m![1], m![1], m![A, B = 4 #{!} 8], m![O]> = input.to_dm(&mut device.tdma);
    let output: DmTensor<i32, m![1], m![1], m![A, B = 4 #{!} 8], m![O]> = device
        .main
        .begin(source.view())
        .fetch::<m![1], m![O]>()
        .collect::<m![1], m![O]>()
        .commit_trim::<m![O]>()
        .commit();
    output.to_hbm(&mut device.tdma)
}

/// Adds one to the first half of every `B` row from a stream that holds every slice, and leaves the
/// second half as it was: only the destination is tiled, so the commit drops the stream's `B >= 4`
/// slices.
#[device(chip = 1, pe = 1)]
pub fn add_one_from_a_whole_stream(
    device: &mut Device,
    input: &HbmTensor<i32, m![1], m![A, B, O]>,
    previous: &HbmTensor<i32, m![1], m![A, B, O]>,
) -> HbmTensor<i32, m![1], m![A, B, O]> {
    let source: DmTensor<i32, m![1], m![1], m![A, B], m![O]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, m![1], m![1], m![A, B], m![O]> = previous.to_dm(&mut device.tdma);
    let destination = output.view_mut().slice_tile::<m![B / 4], 1, m![A, B % 4 #{!} 8]>(0);
    device
        .main
        .begin(source.view())
        .fetch::<m![1], m![O]>()
        .collect::<m![1], m![O]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![O]>()
        .commit_view(destination);
    output.to_hbm(&mut device.tdma)
}

/// Copies a stream that holds every slice into a tensor whose `B >= 4` slices are `Top` padding: the
/// commit writes them with don't-care values.
#[device(chip = 1, pe = 1)]
pub fn copy_a_whole_stream_into_top_padded_slices(
    device: &mut Device,
    input: &HbmTensor<i32, m![1], m![A, B, O]>,
) -> HbmTensor<i32, m![1], m![A, B = 4, O]> {
    let source: DmTensor<i32, m![1], m![1], m![A, B], m![O]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, m![1], m![1], m![A, B = 4 # 8], m![O]> = DmTensor::new();
    device
        .main
        .begin(source.view())
        .fetch::<m![1], m![O]>()
        .collect::<m![1], m![O]>()
        .commit_trim::<m![O]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}

/// Copies the first half of every `B` row at a 4-PE device (1 cluster x 256 slices) and leaves the
/// second half as it was, so the skipped slices lie in every PE.
#[device(chip = 1, pe = 4)]
pub fn copy_into_a_slice_tile_at_four_pes(
    device: &mut Device,
    input: &HbmTensor<i32, m![1], m![E, A, B, O]>,
    previous: &HbmTensor<i32, m![1], m![E, A, B, O]>,
) -> HbmTensor<i32, m![1], m![E, A, B, O]> {
    let source: DmTensor<i32, m![1], m![1], m![E, A, B], m![O]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, m![1], m![1], m![E, A, B], m![O]> = previous.to_dm(&mut device.tdma);
    let destination = output.view_mut().slice_tile::<m![B / 4], 1, m![E, A, B % 4 #{!} 8]>(0);
    device
        .main
        .begin(source.view())
        .fetch::<m![1], m![O]>()
        .collect::<m![1], m![O]>()
        .commit_trim::<m![O]>()
        .commit_view(destination);
    output.to_hbm(&mut device.tdma)
}
