//! Rejection fixture for an unsupported runtime device assertion.

use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32];

/// Transposes `[A, B]` to `[B, A]` with a runtime assertion that compilation must reject.
#[device(chip = 1)]
pub fn runtime_panic(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    for b in 0..<m![B]>::SIZE {
        // A branch result is a runtime scalar even when its condition is not, so the guard
        // survives to lowering instead of folding to `true` and taking the panic with it.
        let bound = if b == 0 { 1000 } else { 1 };
        assert!(b < bound, "runtime guard that must be rejected");
        let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut device.tdma, output_slice);
    }
    output
}
