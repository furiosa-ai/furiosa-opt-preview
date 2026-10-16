//! A transposing pass over one tile of a padded unit axis: each row's largest feature, four rows a
//! packet, for the second half of the rows only.
use furiosa_opt_std::prelude::*;

axes![R = 64, S = 256, H = 16];

type Rows = DmTensor<bf16, m![1], m![1 # 2], m![S], m![R, H]>;
type Largest = DmTensor<bf16, m![1], m![1 # 2], m![S], m![R]>;

/// Rows `32..64` of each slice's largest features; rows `0..32` are left unwritten.
#[device(chip = 1)]
pub fn half_largest(
    device: &mut Device,
    input: &HbmTensor<bf16, m![1], m![S, R, H]>,
) -> HbmTensor<bf16, m![1], m![S, R]> {
    let rows: Rows = input.to_dm(&mut device.tdma);
    let mut largest = Largest::new();
    device
        .main
        .begin(rows.view().tile::<m![R / 32], 1, m![R / 32 = 1 # 2, R % 32, H]>(1))
        .fetch::<m![R / 32 = 1 # 2, R % 32, H / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![R / 32 = 1 # 2, R % 32, H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R / 32 = 1 # 2, R % 32, H / 4], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![R / 32 = 1 # 2, R % 32], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![R / 32 = 1 # 2, R % 32 / 4], m![R % 4 # 16]>()
        .commit_trim::<m![R % 4]>()
        .commit_view(
            largest
                .view_mut()
                .tile::<m![R / 32], 1, m![R / 32 = 1 #{!} 2, R % 32]>(1),
        );
    largest.to_hbm(&mut device.tdma)
}
