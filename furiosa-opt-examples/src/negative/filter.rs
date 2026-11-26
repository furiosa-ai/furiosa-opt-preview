//! Rejection fixtures for the Vector Engine's Filter stage.

use furiosa_opt_std::prelude::*;

axes![A = 8, B = 8, H = 2, K = 16, O = 8, R = 8, F = 8, S = 256];

type Chip = m![1];
type Cluster = m![1];
type Slices = m![A, B];

/// A padded element carries a tag like any other, so a tag filter cannot tell it from data.
#[device(chip = 1, pe = 1)]
pub fn filter_by_tag_over_padded_input(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, K]>,
    previous: &HbmTensor<i32, Chip, m![A, B, H, O]>,
) -> HbmTensor<i32, Chip, m![A, B, H, O]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![K]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, Chip, Cluster, Slices, m![H, O]> = previous.to_dm(&mut device.tdma);
    device
        .main
        .begin(source.view())
        .fetch::<m![K / 4], m![K % 4]>()
        .collect::<m![K / 4], m![K % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False]))
        .vector_filter_by_tag::<m![H], m![O]>(TagGuard::not_matches([
            BitReq::One,
            BitReq::Ignore,
            BitReq::Ignore,
            BitReq::Ignore,
        ]))
        .vector_final()
        .commit_trim::<m![O]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}

/// New axes take the place of the packet's axes only, so the output cannot rename `Time` too.
#[device(chip = 1, pe = 1)]
pub fn filter_by_bitmap_renaming_time(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, K]>,
) -> HbmTensor<i32, Chip, m![A, B, H, O]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![K]> = input.to_dm(&mut device.tdma);
    let packed: DmTensor<i32, Chip, Cluster, Slices, m![H, O]> = device
        .main
        .begin(source.view())
        .fetch::<m![K / 4], m![K % 4]>()
        .collect::<m![K / 4], m![K % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_filter_by_bitmap::<0b0000_1111, m![H], m![O]>()
        .vector_final()
        .commit_trim::<m![O]>()
        .commit();
    packed.to_hbm(&mut device.tdma)
}

/// An empty bitmap keeps no position, so no output layout can describe what the filter writes.
#[device(chip = 1, pe = 1)]
pub fn filter_by_empty_bitmap(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, K]>,
) -> HbmTensor<i32, Chip, m![A, B, K]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![K]> = input.to_dm(&mut device.tdma);
    let kept: DmTensor<i32, Chip, Cluster, Slices, m![K]> = device
        .main
        .begin(source.view())
        .fetch::<m![K / 8], m![K % 8]>()
        .collect::<m![K / 8], m![K % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_filter_by_bitmap::<0, m![K / 8], m![K % 8]>()
        .vector_final()
        .commit_trim::<m![K % 8]>()
        .commit();
    kept.to_hbm(&mut device.tdma)
}

/// Fields 0, 3 and 7 are no `/`, `%` or `=` of `F`, so the output cannot keep the packet's own axis.
#[device(chip = 1, pe = 1)]
pub fn filter_by_bitmap_claiming_the_packet_axis(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, R, F]>,
) -> HbmTensor<i32, Chip, m![A, B, R, F = 3]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![R, F]> = input.to_dm(&mut device.tdma);
    let kept: DmTensor<i32, Chip, Cluster, Slices, m![R, F = 3]> = device
        .main
        .begin(source.view())
        .fetch::<m![R], m![F]>()
        .collect::<m![R], m![F]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_filter_by_bitmap::<0b1000_1001, m![[R, F = 3] / 8], m![[R, F = 3] % 8]>()
        .vector_final()
        .commit_trim::<m![[R, F = 3] % 8]>()
        .commit();
    kept.to_hbm(&mut device.tdma)
}

/// A Slice tile's other slices hold no defined value, so a commit cannot write them where a value is expected.
#[device(chip = 1, pe = 1)]
pub fn filtered_undefined_slice_committed_to_a_valid_slice(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, K]>,
    previous: &HbmTensor<i32, Chip, m![A, B, O]>,
) -> HbmTensor<i32, Chip, m![A, B, O]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![K]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, Chip, Cluster, Slices, m![O]> = previous.to_dm(&mut device.tdma);
    let source = source.view().slice_tile::<m![B / 4], 1, m![A, B % 4 # 8]>(0);
    device
        .main
        .begin(source)
        .fetch::<m![K / 8], m![K % 8]>()
        .collect::<m![K / 8], m![K % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False]))
        .vector_filter_by_tag::<m![1], m![O]>(TagGuard::not_matches([
            BitReq::One,
            BitReq::Ignore,
            BitReq::Ignore,
            BitReq::Ignore,
        ]))
        .vector_final()
        .commit_trim::<m![O]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}

/// A tag filter's undefined slices decide how many packets they emit, so a commit must skip them
/// (`#{!}`) rather than write them as don't-care (`#`): one that emits past its output stops the
/// device.
#[device(chip = 1, pe = 1)]
pub fn filtered_undefined_slice_committed_to_a_dont_care_slice(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A, B, K]>,
    previous: &HbmTensor<i32, Chip, m![A, B = 4, O]>,
) -> HbmTensor<i32, Chip, m![A, B = 4, O]> {
    let source: DmTensor<i32, Chip, Cluster, Slices, m![K]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, Chip, Cluster, m![A, B = 4 # 8], m![O]> = previous.to_dm(&mut device.tdma);
    let source = source.view().slice_tile::<m![B / 4], 1, m![A, B % 4 # 8]>(0);
    device
        .main
        .begin(source)
        .fetch::<m![K / 8], m![K % 8]>()
        .collect::<m![K / 8], m![K % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False]))
        .vector_filter_by_tag::<m![1], m![O]>(TagGuard::not_matches([
            BitReq::One,
            BitReq::Ignore,
            BitReq::Ignore,
            BitReq::Ignore,
        ]))
        .vector_final()
        .commit_trim::<m![O]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}

/// Cluster 1 is padding, so its undefined values may pass the guard, and a commit cannot skip a
/// cluster: an output of fewer packets than a slice reads could overflow there and stop the device.
#[device(chip = 1)]
pub fn filter_by_tag_on_a_padded_cluster(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![S, K]>,
    previous: &HbmTensor<i32, Chip, m![S, O]>,
) -> HbmTensor<i32, Chip, m![S, O]> {
    let source: DmTensor<i32, Chip, m![1 # 2], m![S], m![K]> = input.to_dm(&mut device.tdma);
    let mut output: DmTensor<i32, Chip, m![1 # 2], m![S], m![O]> = previous.to_dm(&mut device.tdma);
    device
        .main
        .begin(source.view())
        .fetch::<m![K / 8], m![K % 8]>()
        .collect::<m![K / 8], m![K % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False]))
        .vector_filter_by_tag::<m![1], m![O]>(TagGuard::not_matches([
            BitReq::One,
            BitReq::Ignore,
            BitReq::Ignore,
            BitReq::Ignore,
        ]))
        .vector_final()
        .commit_trim::<m![O]>()
        .commit_view(output.view_mut());
    output.to_hbm(&mut device.tdma)
}
