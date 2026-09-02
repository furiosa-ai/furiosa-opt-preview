//! Device-translation rejection fixtures for `#[unroll]`.

use furiosa_opt_std::prelude::*;

use crate::unroll_loop::transpose_row;
pub use crate::unroll_loop::{A, B};
/// A runtime trip count, which cannot be expanded into a static number of copies.
#[device(chip = 1)]
pub fn unroll_dynamic_count(
    ctx: &mut Context,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    for outer in 0..2 {
        let rows: usize = if outer == 0 { 2 } else { 4 };
        #[unroll]
        for b in 0..rows {
            transpose_row(ctx, &input, &mut output, b);
        }
    }
    output
}

/// An empty loop range.
#[device(chip = 1)]
pub fn unroll_empty_range(
    ctx: &mut Context,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    #[unroll]
    #[expect(clippy::reversed_empty_ranges, reason = "the empty range is the fixture")]
    for b in 3..1 {
        transpose_row(ctx, &input, &mut output, b);
    }
    output
}
