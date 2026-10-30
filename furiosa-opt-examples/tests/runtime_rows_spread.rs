use furiosa_opt_examples::runtime_rows_spread::{
    Bank, Group, Input, Kp, Lane, One, Out, Output, Red, Weight, runtime_rows_spread,
};
use furiosa_opt_std::prelude::*;

const SENTINEL: f32 = -9876.0;

/// A small integer in {-1, 0, 1} for element `i`, so every sum is exact in f32 and bf16.
fn value(i: usize) -> f32 {
    f32::from(i16::try_from((i * 7 + i / 13) % 3).unwrap()) - 1.0
}

/// Live row groups of a contraction whose reduce spreads the row groups over slices equal an
/// exact host oracle; the slices of the groups past the count keep their old value.
#[tokio::test]
async fn runtime_rows_spread_preserve_tail() -> eyre::Result<()> {
    let mut ctx = Device::new(runtime_rows_spread.topology())?;
    let input = HostTensor::<bf16, Input>::from_vec((0..<Input as M>::SIZE).map(|i| bf16::from_f32(value(i))))
        .to_hbm::<m![1], Input>(&mut ctx.pdma)
        .await?;
    let weight = HostTensor::<bf16, Weight>::from_vec((0..<Weight as M>::SIZE).map(|i| bf16::from_f32(value(i + 5))))
        .to_hbm::<m![1], Weight>(&mut ctx.pdma)
        .await?;
    let expected = |b: usize, g: usize, l: usize, o: usize| -> f32 {
        let mut sum = 0.0;
        for k in 0..Kp::SIZE {
            for r in 0..Red::SIZE {
                let a = (((b * Kp::SIZE + k) * Group::SIZE + g) * Lane::SIZE + l) * Red::SIZE + r;
                let w = ((b * Kp::SIZE + k) * Out::SIZE + o) * Red::SIZE + r;
                sum += value(a) * value(w + 5);
            }
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
        launch(runtime_rows_spread, (&mut ctx, &count, &input, &weight, &mut output)).await?;
        let output = output.to_host::<Output>(&mut ctx.pdma).await?.into_vec();
        let live = usize::try_from(r).unwrap_or(0).min(Group::SIZE);
        for b in 0..Bank::SIZE {
            for g in 0..Group::SIZE {
                for l in 0..Lane::SIZE {
                    for o in 0..Out::SIZE {
                        let oracle = bf16::from_f32(expected(b, g, l, o));
                        let want = if g < live { oracle } else { bf16::from_f32(SENTINEL) };
                        let got = output[at(b, g, o, l)];
                        assert_eq!(got.to_bits(), want.to_bits(), "r={r} b={b} g={g} l={l} o={o}");
                    }
                }
            }
        }
    }
    Ok(())
}
