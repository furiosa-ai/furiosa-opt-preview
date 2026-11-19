//! Rejection fixtures for `#[bound(MAX)]`.
//!
//! A `MAX` known only at run time is a doctest rather than a kernel, for the reason
//! [`negative`](crate::negative) gives: rustc refuses it, since the marker reads `MAX` in a `const`
//! block.
//!
//! ```compile_fail,E0435
//! # #![feature(proc_macro_hygiene, stmt_expr_attributes)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! fn count(end: usize, max: usize) {
//!     #[bound(max)]
//!     for _ in 0..end {}
//! }
//! ```
//!
//! The same loop with a constant `MAX` compiles, so the one above fails for `MAX` and nothing else.
//!
//! ```
//! # #![feature(proc_macro_hygiene, stmt_expr_attributes)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! fn count(end: usize) {
//!     #[bound(4)]
//!     for _ in 0..end {}
//! }
//! ```

use furiosa_opt_std::prelude::*;

use crate::bound_loop::increment;
pub use crate::bound_loop::{Slice, Width};

type Row = DmTensor<i32, m![1], m![1], m![Slice], m![Width]>;
type Tensor = HbmTensor<i32, m![1], m![Slice, Width]>;

/// A constant end of two differs from the declared bound three.
#[device(chip = 1, pe = 1)]
pub fn constant_end_mismatches_bound(device: &mut Device, input: &Tensor) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    #[bound(3)]
    for _ in 0..2 {
        increment(device, &mut row);
    }
    row.to_hbm(&mut device.tdma)
}

/// A declared bound beyond the `i32` index range.
#[device(chip = 1, pe = 1)]
pub fn bound_exceeds_index(device: &mut Device, input: &Tensor, count: &HbmScalar<i32>) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    let count = count.to_spm(&mut device.tdma);
    #[bound(i32::MAX as usize + 1)]
    for _ in 0..count {
        increment(device, &mut row);
    }
    row.to_hbm(&mut device.tdma)
}

/// A runtime loop end requires an explicit bound.
#[device(chip = 1, pe = 1)]
pub fn missing_bound(device: &mut Device, input: &Tensor, count: &HbmScalar<i32>) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    let count = count.to_spm(&mut device.tdma);
    for _ in 0..count {
        increment(device, &mut row);
    }
    row.to_hbm(&mut device.tdma)
}

/// A declared bound of zero, though a bound must be at least one.
#[device(chip = 1, pe = 1)]
pub fn zero_bound(device: &mut Device, input: &Tensor, count: &HbmScalar<i32>) -> Tensor {
    let mut row: Row = input.to_dm(&mut device.tdma);
    let count = count.to_spm(&mut device.tdma);
    #[bound(0)]
    for _ in 0..count {
        increment(device, &mut row);
    }
    row.to_hbm(&mut device.tdma)
}
