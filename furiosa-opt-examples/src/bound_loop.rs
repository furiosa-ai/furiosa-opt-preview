//! Loops bounded by `#[bound(MAX)]`, whose end may be known only at run time.

use furiosa_opt_std::prelude::*;

axes![Slice = 64, Width = 8];

type Row = DmTensor<i32, m![1], m![1], m![Slice], m![Width]>;
type Tensor = HbmTensor<i32, m![1], m![Slice, Width]>;

/// Adds a constant to every element and returns a new DM tensor.
fn add_const(device: &mut Device, row: &Row, amount: i32) -> Row {
    device
        .main
        .begin(row.view())
        .fetch::<m![1], m![Width]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![Width]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, amount)
        .vector_final()
        .commit_trim::<m![Width]>()
        .commit()
}

/// Adds one to every element of `row` in place.
pub(crate) fn increment(device: &mut Device, row: &mut Row) {
    let next = add_const(device, row, 1);
    next.view().to_dm_view(&mut device.tdma, row.view_mut());
}

/// Adds `count.max(0)` to every input element for `count <= 4`.
#[device(chip = 1, pe = 1)]
pub fn loop_with_runtime_end(device: &mut Device, input: &Tensor, count: &HbmScalar<i32>) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    let count = count.to_spm(&mut device.tdma);
    #[bound(4)]
    for _ in 0..count {
        increment(device, &mut row);
    }
    row.to_hbm(&mut device.tdma)
}

/// Adds `((8 - s % 8) % 8).min(rows - s)` to each element before the next tile boundary.
/// The start is `s = rows.min(offset)`.
#[device(chip = 1, pe = 1)]
pub fn increment_to_tile_boundary(
    device: &mut Device,
    input: &Tensor,
    rows: &HbmScalar<i32>,
    offset: &HbmScalar<i32>,
) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    let rows = rows.to_spm(&mut device.tdma) as usize;
    let offset = offset.to_spm(&mut device.tdma) as usize;
    // Restrict the requested offset to the live rows, including an empty span.
    let start = rows.min(offset);
    let remaining = rows - start;
    // Rows before the next tile boundary, so fewer than the rows of one tile.
    type Tile = m![8];
    let first = ((Tile::SIZE - start % Tile::SIZE) % Tile::SIZE).min(remaining);
    increment_within_tile::<Tile>(device, &mut row, first);
    row.to_hbm(&mut device.tdma)
}

/// Adds one to `row` `count` times, for `count < Tile::SIZE`.
fn increment_within_tile<Tile: M>(device: &mut Device, row: &mut Row, count: usize) {
    #[bound(Tile::SIZE - 1)]
    for _ in 0..count {
        increment(device, row);
    }
}

/// Runs nested bounded loops, producing `input + count * (count - 1) / 2` for `0 <= count <= 4`.
#[device(chip = 1, pe = 1)]
pub fn nested_dynamic_loops(device: &mut Device, input: &Tensor, count: &HbmScalar<i32>) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    let count = count.to_spm(&mut device.tdma) as usize;
    #[bound(4)]
    for i in 0..count {
        // The outer bound keeps `i` within the inner bound 3.
        #[bound(3)]
        for _ in 0..i {
            increment(device, &mut row);
        }
    }
    row.to_hbm(&mut device.tdma)
}

/// Reuses the runtime end inside the loop, producing `input + count * count` for `0 <= count <= 4`.
#[device(chip = 1, pe = 1)]
pub fn runtime_end_reused_in_body(device: &mut Device, input: &Tensor, count: &HbmScalar<i32>) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    let count = count.to_spm(&mut device.tdma) as usize;
    #[bound(4)]
    for _ in 0..count {
        increment_times(device, &mut row, count);
    }
    row.to_hbm(&mut device.tdma)
}

/// Adds one to `row` `count` times, for `count <= 4`.
fn increment_times(device: &mut Device, row: &mut Row, count: usize) {
    #[bound(4)]
    for _ in 0..count {
        increment(device, row);
    }
}

/// Runs a constant end equal to its bound, producing `input + 2`.
#[device(chip = 1, pe = 1)]
pub fn constant_end_matches_bound(device: &mut Device, input: &Tensor) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    #[bound(2)]
    for _ in 0..2 {
        increment(device, &mut row);
    }
    row.to_hbm(&mut device.tdma)
}
