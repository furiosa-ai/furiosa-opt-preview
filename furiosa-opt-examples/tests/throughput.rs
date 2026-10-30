//! Build-only root with many independent specializations for link throughput measurements.

use furiosa_opt_examples::generic::{B, bundle_copy};
use furiosa_opt_std::prelude::*;

macro_rules! copies {
    ($($axis:ident = $len:literal),* $(,)?) => {
        axes![$($axis = $len),*];

        fn loads() -> Result<(), furiosa_opt_std::Error> {
            let device: Device = std::hint::black_box(None).unwrap();
            $(
                Loadable::<(&mut Device, &HbmTensor<i32, m![1], m![B, $axis]>)>::load(bundle_copy, &device)?;
            )*
            Ok(())
        }
    };
}

copies![
    A1 = 16,
    A2 = 32,
    A3 = 48,
    A4 = 64,
    A5 = 80,
    A6 = 96,
    A7 = 112,
    A8 = 128,
    A9 = 144,
    A10 = 160,
    A11 = 176,
    A12 = 192,
    A13 = 208,
    A14 = 224,
    A15 = 240,
    A16 = 256,
    A17 = 272,
    A18 = 288,
    A19 = 304,
    A20 = 320,
    A21 = 336,
    A22 = 352,
    A23 = 368,
    A24 = 384,
    A25 = 400,
    A26 = 416,
    A27 = 432,
    A28 = 448,
    A29 = 464,
    A30 = 480,
    A31 = 496,
    A32 = 512,
];

/// All specializations remain reachable from a build-only harness.
#[test]
fn registers_specializations() {
    if std::hint::black_box(false) {
        loads().unwrap();
    }
}
