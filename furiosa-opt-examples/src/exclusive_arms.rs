//! A runtime `if` whose arms each land one tensor: no run issues both transfers, so they can share
//! a completion id.

use furiosa_opt_std::prelude::*;

axes![One = 1, Rows = 8, W = 1024];

type Chip = m![1];
type Row = DmTensor<i32, Chip, m![1], m![W / 4], m![W % 4]>;

/// `long` when more than four rows are live, else `short`. Oracle: the input it names.
#[device(chip = 1, pe = 4)]
pub fn exclusive_loads(
    device: &mut Device,
    rows: &HbmTensor<i32, Chip, m![One]>,
    long: &HbmTensor<i32, Chip, m![W]>,
    short: &HbmTensor<i32, Chip, m![W]>,
) -> HbmTensor<i32, Chip, m![W]> {
    let mut landed = Row::new();
    if rows.extent::<m![Rows]>() > 4 {
        long.view().to_dm_view(&mut device.tdma, landed.view_mut());
    } else {
        short.view().to_dm_view(&mut device.tdma, landed.view_mut());
    }
    landed.to_hbm(&mut device.tdma)
}
