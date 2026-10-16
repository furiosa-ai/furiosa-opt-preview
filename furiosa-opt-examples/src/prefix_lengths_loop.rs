//! A prefix gather a trip, each limited by its trip's tile of one host length table.

use furiosa_opt_std::prelude::*;

axes![Trips = 4, Entries = 8, Width = 256, Key = 2];

type Chip = m![1];

#[device(chip = 1)]
pub fn prefix_lengths_loop(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![Entries, Width]>,
    indices: &HbmTensor<i32, Chip, m![Key]>,
    lengths: &HbmTensor<i32, Chip, m![Trips]>,
    output: &mut HbmTensor<bf16, Chip, m![Trips, Key, Width]>,
) {
    for trip in 0..Trips::SIZE {
        // SAFETY: the tile is the trip's one length, and nothing writes the table meanwhile.
        let length: HbmTensor<i32, Chip, m![1]> =
            unsafe { lengths.view().tile::<m![Trips], 1, m![Trips = 1 # Trips]>(trip).alias() };
        let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Key, Width]> = table
            .gather::<m![Entries], m![Width]>()
            .by_byte_offsets(indices)
            .prefix::<m![Key], _>(&length, 0)
            .to_dm(&mut device.tdma);
        rows.view().to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![Trips], 1, m![Trips = 1 #{!} Trips, Key, Width]>(trip),
        );
    }
}
