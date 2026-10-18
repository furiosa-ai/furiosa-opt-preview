use super::*;

/// A flit's place in its span of 16 flits, the first and last marked by a log pass.
fn span_ends(
    device: &mut Device,
    position: &HbmTensor<i32, Chip, m![A]>,
) -> VrfTensor<i4, Chip, Cluster, Slice, m![A]> {
    let position = position.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);
    device
        .main
        .begin(position.view())
        .fetch::<m![A / 8], m![A % 8]>()
        .collect::<m![A / 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Equal(0),
            Cmp::Equal(15),
            Cmp::False,
            Cmp::False,
        ]))
        .vector_log(&mut device.sub)
}

/// Each lane's prefix sum over every span of 16 flits of a row (`position` = a flit's place in its
/// span), the carry in the operand register: a span's first flit reads none, its last writes none.
#[device(chip = 1)]
pub fn ve_cumsum_spans_i32(
    device: &mut Device,
    position: &HbmTensor<i32, Chip, m![A]>,
    input: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let ends = span_ends(device, position);
    let input = input.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);
    let result: DmTensor<i32, Chip, Cluster, Slice, m![A]> = device
        .main
        .begin(input.view())
        .fetch::<m![A / 8], m![A % 8]>()
        .collect::<m![A / 8], m![A % 8]>()
        .vector_init()
        .vector_tagged(&ends)
        .vector_cumsum(
            ClipBinaryOpI32::AddFxpSat,
            TagGuard::not_matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]),
            TagGuard::not_matches([BitReq::Ignore, BitReq::One, BitReq::Ignore, BitReq::Ignore]),
        )
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}

/// The f32 form of [`ve_cumsum_spans_i32`].
#[device(chip = 1)]
pub fn ve_cumsum_spans_f32(
    device: &mut Device,
    position: &HbmTensor<i32, Chip, m![A]>,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let ends = span_ends(device, position);
    let input = input.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);
    let result: DmTensor<f32, Chip, Cluster, Slice, m![A]> = device
        .main
        .begin(input.view())
        .fetch::<m![A / 8], m![A % 8]>()
        .collect::<m![A / 8], m![A % 8]>()
        .vector_init()
        .vector_tagged(&ends)
        .vector_cumsum(
            ClipBinaryOpF32::Add,
            TagGuard::not_matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]),
            TagGuard::not_matches([BitReq::Ignore, BitReq::One, BitReq::Ignore, BitReq::Ignore]),
        )
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit();
    result.to_hbm(&mut device.tdma)
}
