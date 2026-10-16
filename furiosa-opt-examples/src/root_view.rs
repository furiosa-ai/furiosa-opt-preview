//! Indexed DMA addressed relative to a per-iteration window of a larger HBM tensor.
//!
//! A model forward keeps every layer's state in one root tensor and selects a layer inside a
//! runtime loop, so gathers and scatters must take their base from the selected window rather
//! than from the start of the allocation.

use furiosa_opt_std::prelude::*;

axes![Layers = 3, Entries = 4, Width = 256, Key = 2];

type Chip = m![1];

/// Gathers `Key` rows from each layer's window of `root`, one layer per loop iteration.
#[device(chip = 1)]
pub fn gather_layer_rows(
    device: &mut Device,
    root: &HbmTensor<bf16, Chip, m![Layers, Entries, Width]>,
    indices: &HbmTensor<i32, Chip, m![Key]>,
    lengths: &HbmTensor<i32, Chip, m![1]>,
    count: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![Layers, Key, Width]>,
) {
    for layer in 0..count.extent::<m![Layers]>() {
        let window = root
            .view()
            .tile::<m![Layers], 1, m![Layers = 1 # Layers, Entries, Width]>(layer);
        let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = window
            .gather::<m![Entries], m![Layers = 1 # Layers, Width]>()
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

/// Scatters each layer's `Key` rows into that layer's four-entry block of `table`.
#[device(chip = 1)]
pub fn scatter_layer_rows(
    device: &mut Device,
    values: &HbmTensor<bf16, Chip, m![Layers, Key, Width]>,
    indices: &HbmTensor<i32, Chip, m![Key]>,
    count: &HbmTensor<i32, Chip, m![1]>,
    table: &mut Slice<HbmTensor<bf16, Chip, m![Width]>>,
) {
    for layer in 0..count.extent::<m![Layers]>() {
        let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = values
            .view()
            .tile::<m![Layers], 1, m![Layers = 1 # Layers, Key, Width]>(layer)
            .to_dm(&mut device.tdma);
        table
            .scatter::<m![1], m![1 # 2, 1 # 256, Width]>()
            .entry(layer * Entries::SIZE)
            .by_byte_offsets(indices.view())
            .from_dm(&mut device.tdma, rows);
    }
}
