use furiosa_opt_examples::runtime_rows_scaled::{
    Bank, Group, Input, Kp, Lane, One, Out, Output, Red, Scale, Weight, runtime_rows_scaled, runtime_rows_scaled_single,
};
use furiosa_opt_std::prelude::*;

const SENTINEL: f32 = -9876.0;

/// A small integer in {-1, 0, 1} for element `i`, so every product and sum is exact in f32.
fn value(i: usize) -> f32 {
    f32::from(i16::try_from((i * 7 + i / 13) % 3).unwrap()) - 1.0
}

/// A power of two in {1/2, 1, 2, 4} for scale element `i`, distinct across neighbouring rows.
fn scale(i: usize) -> f32 {
    [0.5, 1.0, 2.0, 4.0][(i * 5 + i / 8) % 4]
}

/// Live row groups of a counted contraction that scales each slice's partial sums by its own
/// rows' scales before the slices add them equal an exact host oracle; dead groups keep their
/// old value.
macro_rules! preserves_tail {
    ($test:ident, $kernel:ident) => {
        #[tokio::test]
        async fn $test() -> eyre::Result<()> {
            let mut ctx = Device::new($kernel.topology())?;
            let input = HostTensor::<bf16, Input>::from_vec((0..<Input as M>::SIZE).map(|i| bf16::from_f32(value(i))))
                .to_hbm::<m![1], Input>(&mut ctx.pdma)
                .await?;
            let weight =
                HostTensor::<bf16, Weight>::from_vec((0..<Weight as M>::SIZE).map(|i| bf16::from_f32(value(i + 5))))
                    .to_hbm::<m![1], Weight>(&mut ctx.pdma)
                    .await?;
            let scales = HostTensor::<f32, Scale>::from_vec((0..<Scale as M>::SIZE).map(scale))
                .to_hbm::<m![1], Scale>(&mut ctx.pdma)
                .await?;
            let expected = |b: usize, g: usize, l: usize, o: usize| -> f32 {
                let mut sum = 0.0;
                for k in 0..Kp::SIZE {
                    let row = ((b * Kp::SIZE + k) * Group::SIZE + g) * Lane::SIZE + l;
                    let mut partial = 0.0;
                    for r in 0..Red::SIZE {
                        let w = ((b * Kp::SIZE + k) * Out::SIZE + o) * Red::SIZE + r;
                        partial += value(row * Red::SIZE + r) * value(w + 5);
                    }
                    sum += partial * scale(row);
                }
                sum
            };
            let at = |b: usize, g: usize, o: usize, l: usize| {
                (((b * Group::SIZE + g) * (Out::SIZE / 4) + o / 4) * Lane::SIZE + l) * 4 + o % 4
            };
            let sentinel = vec![bf16::from_f32(SENTINEL); <Output as M>::SIZE];
            for r in [16, -1, 0, 1, 7, 8, 15, 17] {
                let count = HostTensor::<i32, m![One]>::from_vec([r])
                    .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
                    .await?;
                let mut output = HostTensor::<bf16, Output>::from_vec(sentinel.clone())
                    .to_hbm::<m![1], Output>(&mut ctx.pdma)
                    .await?;
                launch($kernel, (&mut ctx, &count, &input, &weight, &scales, &mut output)).await?;
                let output = output.to_host::<Output>(&mut ctx.pdma).await?.into_vec();
                let live = usize::try_from(r).unwrap_or(0).min(Group::SIZE);
                for b in 0..Bank::SIZE {
                    for g in 0..Group::SIZE {
                        for l in 0..Lane::SIZE {
                            for o in 0..Out::SIZE {
                                let oracle = bf16::from_f32(expected(b, g, l, o));
                                let want = if g < live {
                                    oracle
                                } else {
                                    bf16::from_f32(SENTINEL)
                                };
                                let got = output[at(b, g, o, l)];
                                assert_eq!(got.to_bits(), want.to_bits(), "r={r} b={b} g={g} l={l} o={o}");
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

preserves_tail!(runtime_rows_scaled_preserve_tail, runtime_rows_scaled);
preserves_tail!(runtime_rows_scaled_single_preserve_tail, runtime_rows_scaled_single);
