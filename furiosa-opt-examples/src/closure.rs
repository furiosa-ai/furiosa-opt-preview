//! Closures retain their captures and forward device arguments through direct and `FnOnce` calls.

use furiosa_opt_std::prelude::*;

axes![Rows = 512, Cols = 32];
type Hbm = HbmTensor<i32, m![1], m![Rows, Cols]>;

/// Copies two captured inputs into the corresponding closure arguments.
#[device(chip = 1)]
pub fn pass_device(
    device: &mut Device,
    first_input: &Hbm,
    second_input: &Hbm,
    first_output: &mut Hbm,
    second_output: &mut Hbm,
) {
    let copy = |device: &mut Device, first_output: &mut Hbm, second_output: &mut Hbm| {
        first_input
            .view()
            .to_hbm_view(&mut device.tdma, first_output.view_mut());
        second_input
            .view()
            .to_hbm_view(&mut device.tdma, second_output.view_mut());
    };
    copy(device, first_output, second_output);
}

fn lend(device: &mut Device, run: impl FnOnce(&mut Device)) {
    run(device)
}

/// The inline closure takes `FnOnce` from the bound and is called without an adapter.
#[device(chip = 1)]
pub fn lend_device(device: &mut Device, input: &Hbm, output: &mut Hbm) {
    lend(device, |device| {
        input.view().to_hbm_view(&mut device.tdma, output.view_mut())
    });
}

/// Unlike [`lend_device`], the stored closure is inferred as `FnMut` and needs a `FnOnce` adapter.
#[device(chip = 1)]
pub fn lend_stored(device: &mut Device, input: &Hbm, output: &mut Hbm) {
    let copy = |device: &mut Device| input.view().to_hbm_view(&mut device.tdma, output.view_mut());
    lend(device, copy);
}
