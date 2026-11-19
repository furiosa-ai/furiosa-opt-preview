//! Control-flow shaping primitives.

use furiosa_opt_macro::primitive;

/// Carries `#[unroll]` from macro expansion to the device translator.
/// The marker is a call because rustc does not preserve loop attributes in MIR.
#[doc(hidden)]
#[primitive(loop_hint::unroll)]
pub fn __loop_hint_unroll() {}

/// Carries the `MAX` of `#[bound(MAX)]` from macro expansion to the device translator. On the CPU
/// backend it also checks the loop index, so a loop ending above `MAX` panics at index `MAX`.
#[doc(hidden)]
#[primitive(loop_hint::bound)]
#[track_caller]
pub fn __loop_hint_bound<I: TryInto<usize>>(index: I, max: usize) {
    assert!(
        index.try_into().is_ok_and(|index| index < max),
        "the end of a `#[bound({max})]` loop exceeds {max}"
    );
}
