//! Runtime loops carrying tensor state between iterations.
//!
//! Tensor reassignment and mutable-view writes preserve the updated values for the next
//! iteration and for consumers after the loop. A tensor that a reassigned reference first
//! referred to keeps its value.

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

/// Copies `row` into a new DM tensor and adds a constant to every element of the copy.
fn add_const_to_copy(device: &mut Device, row: &Row, amount: i32) -> Row {
    let copy: Row = row.to_dm(&mut device.tdma);
    add_const(device, &copy, amount)
}

/// Reassigns the carried tensor, then updates the new tensor through a mutable view.
/// Each of three iterations adds three.
///
/// Oracle: `output == input + 9`.
#[device(chip = 1, pe = 1)]
pub fn loop_reassign_then_write(device: &mut Device, input: &Tensor) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    for _ in 0..3 {
        row = add_const_to_copy(device, &row, 1);
        let updated = add_const(device, &row, 2);
        updated.view().to_dm_view(&mut device.tdma, row.view_mut());
    }
    row.to_hbm(&mut device.tdma)
}

/// Reassigns a reference that first refers to `original`, then reads both after the loop.
/// Each of three iterations points the reference at a new tensor one greater than the last.
///
/// Oracle: `outputs == (input, input + 3)`.
#[device(chip = 1, pe = 1)]
pub fn loop_reassign_ref_keeps_original(device: &mut Device, input: &Tensor) -> (Tensor, Tensor) {
    let original: Row = input.to_dm(&mut device.tdma);
    let mut current = &original;
    let mut next: std::mem::ManuallyDrop<Row>;
    #[unroll]
    for _ in 0..3 {
        next = std::mem::ManuallyDrop::new(add_const(device, current, 1));
        current = &next;
    }
    (original.to_hbm(&mut device.tdma), current.to_hbm(&mut device.tdma))
}
