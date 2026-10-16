//! Device only: the CPU `commit_prefix` tiles the destination by the row digit, which a write tile
//! padding that digit's own axis does not admit as one strided run.
#![cfg(backend = "npu")]
use furiosa_opt_examples::runtime_rows_tiles::{
    Bank, Input, Kp, One, Out, Output, Red, Rows, Weight, runtime_rows_tiles,
};
use furiosa_opt_std::prelude::*;

const SENTINEL: f32 = -9876.0;

/// A small integer in {-1, 0, 1} for element `i`, so every sum is exact in f32 and bf16.
fn value(i: usize) -> f32 {
    f32::from(i16::try_from((i * 7 + i / 13) % 3).unwrap()) - 1.0
}

/// Each wave commits its live 32-row quarters into its write tile, equal to an exact host oracle;
/// the quarters past the count keep their old value in both waves.
#[tokio::test]
async fn runtime_rows_tiles_preserve_tail() -> eyre::Result<()> {
    let mut ctx = Device::new(runtime_rows_tiles.topology())?;
    let input = HostTensor::<bf16, Input>::from_vec((0..<Input as M>::SIZE).map(|i| bf16::from_f32(value(i))))
        .to_hbm::<m![1], Input>(&mut ctx.pdma)
        .await?;
    let weight = HostTensor::<bf16, Weight>::from_vec((0..<Weight as M>::SIZE).map(|i| bf16::from_f32(value(i + 5))))
        .to_hbm::<m![1], Weight>(&mut ctx.pdma)
        .await?;
    let expected = |b: usize, row: usize, o: usize| -> f32 {
        let mut sum = 0.0;
        for k in 0..Kp::SIZE {
            for r in 0..Red::SIZE {
                let a = ((b * Kp::SIZE + k) * Rows::SIZE + row) * Red::SIZE + r;
                let w = ((b * Kp::SIZE + k) * Out::SIZE + o) * Red::SIZE + r;
                sum += value(a) * value(w + 5);
            }
        }
        sum
    };
    let at = |b: usize, row: usize, o: usize| ((b * (Out::SIZE / 8) + o / 8) * Rows::SIZE + row) * 8 + o % 8;
    for r in [-1, 0, 1, 31, 32, 33, 127, 128, 129, 200, 256, 300] {
        let count = HostTensor::<i32, m![One]>::from_vec([r])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await?;
        let mut output = HostTensor::<bf16, Output>::from_vec(vec![bf16::from_f32(SENTINEL); <Output as M>::SIZE])
            .to_hbm::<m![1], Output>(&mut ctx.pdma)
            .await?;
        launch(runtime_rows_tiles, (&mut ctx, &count, &input, &weight, &mut output)).await?;
        let output = output.to_host::<Output>(&mut ctx.pdma).await?.into_vec();
        let quarters = usize::try_from(r).unwrap_or(0).div_ceil(32);
        for b in 0..Bank::SIZE {
            for row in 0..Rows::SIZE {
                let live = (row % 128) / 32 < quarters.saturating_sub(row / 128 * 4).min(4);
                for o in 0..Out::SIZE {
                    let want = if live {
                        bf16::from_f32(expected(b, row, o))
                    } else {
                        bf16::from_f32(SENTINEL)
                    };
                    assert_eq!(
                        output[at(b, row, o)].to_bits(),
                        want.to_bits(),
                        "r={r} b={b} row={row} o={o}"
                    );
                }
            }
        }
    }
    Ok(())
}
