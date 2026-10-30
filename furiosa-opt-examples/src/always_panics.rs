//! Device functions every run of which panics, as a calculation that is not written yet does. The
//! compiler stores the panic in place of an image, and loading the function raises it on the host.

use furiosa_opt_std::prelude::*;

axes![A = 8, X = 32];

type Chip = m![1];
type Tensor = HbmTensor<bf16, Chip, m![A, X]>;

fn pending(_: &Tensor) -> Tensor {
    todo!()
}

/// Panics before it can return.
#[device(chip = 1)]
pub fn always_panics(device: &mut Device, x: &Tensor) -> Tensor {
    let _ = device;
    pending(x)
}

/// Panics in the first trip of a loop whose trip count is a constant.
#[device(chip = 1)]
pub fn panics_in_loop(device: &mut Device, x: &Tensor) -> Tensor {
    let _ = device;
    #[unroll]
    for _ in 0..2 {
        let _ = pending(x);
    }
    pending(x)
}
