//! Per-layer paged state addressed by a rolled device loop.

use furiosa_opt_std::prelude::*;

axes![Layers = 4, Entries = 4, Width = 256, Key = 2];
type Chip = m![1];
pub type Block = HbmTensor<bf16, Chip, m![Width]>;

/// Writes and reads each layer's pool with layer-local byte offsets.
#[device(chip = 1)]
pub fn forward(
    device: &mut Device,
    pools: &mut Slice<Block>,
    values: &HbmTensor<bf16, Chip, m![Layers, Key, Width]>,
    indices: &HbmTensor<i32, Chip, m![Key]>,
    lengths: &HbmTensor<i32, Chip, m![1]>,
    count: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![Layers, Key, Width]>,
) {
    for layer in 0..count.extent::<m![Layers]>() {
        let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = values
            .view()
            .tile::<m![Layers], 1, m![Layers = 1 # Layers, Key, Width]>(layer)
            .to_dm(&mut device.tdma);
        pools
            .scatter::<m![1], m![1 # 2, 1 # 256, Width]>()
            .entry(layer * Entries::SIZE)
            .by_byte_offsets(indices.view())
            .from_dm(&mut device.tdma, rows);
        let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = pools
            .gather::<m![1], m![Width]>()
            .entry(layer * Entries::SIZE)
            .by_byte_offsets(indices)
            .prefix::<m![Key], _>(lengths, 0)
            .to_dm(&mut device.tdma);
        rows.view().to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![Layers], 1, m![Layers = 1 #{!} Layers, Key, Width]>(layer),
        );
    }
}

/// The constant-entry control for the same per-layer writes and reads.
#[device(chip = 1)]
pub fn unrolled(
    device: &mut Device,
    pools: &mut Slice<Block>,
    values: &HbmTensor<bf16, Chip, m![Layers, Key, Width]>,
    indices: &HbmTensor<i32, Chip, m![Key]>,
    lengths: &HbmTensor<i32, Chip, m![1]>,
    count: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![Layers, Key, Width]>,
) {
    let count = count.extent::<m![Layers]>();
    #[unroll]
    for layer in 0..Layers::SIZE {
        if layer < count {
            let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = values
                .view()
                .tile::<m![Layers], 1, m![Layers = 1 # Layers, Key, Width]>(layer)
                .to_dm(&mut device.tdma);
            pools
                .scatter::<m![1], m![1 # 2, 1 # 256, Width]>()
                .entry(layer * Entries::SIZE)
                .by_byte_offsets(indices.view())
                .from_dm(&mut device.tdma, rows);
            let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = pools
                .gather::<m![1], m![Width]>()
                .entry(layer * Entries::SIZE)
                .by_byte_offsets(indices)
                .prefix::<m![Key], _>(lengths, 0)
                .to_dm(&mut device.tdma);
            rows.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![Layers], 1, m![Layers = 1 #{!} Layers, Key, Width]>(layer),
            );
        }
    }
}
