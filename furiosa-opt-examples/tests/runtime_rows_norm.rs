use furiosa_opt_examples::runtime_rows_norm::{Hidden, One, Rows, X, runtime_rows_norm};
use furiosa_opt_std::prelude::*;

type Output = m![X, Rows, 1 # 8];

/// A counted reduction across slices writes each live row's exact sum on every slice and leaves
/// the rows past the count untouched.
#[tokio::test]
async fn runtime_rows_norm_preserve_tail() -> eyre::Result<()> {
    let mut ctx = Device::new(runtime_rows_norm.topology())?;
    let value = |row: usize, k: usize| i16::try_from((row * Hidden::SIZE + k) % 17).unwrap() - 8;
    let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(
        (0..Rows::SIZE * Hidden::SIZE).map(|i| bf16::from_f32(f32::from(value(i / Hidden::SIZE, i % Hidden::SIZE)))),
    )
    .to_hbm::<m![1], m![Rows, Hidden]>(&mut ctx.pdma)
    .await?;
    let expected = |row: usize| -> f32 { (0..Hidden::SIZE).map(|k| f32::from(value(row, k).pow(2))).sum() };
    let cells = X::SIZE * Rows::SIZE * 8;
    for r in [-1, 0, 1, 7, 8, 64, 127, 128, 129] {
        let count = HostTensor::<i32, m![One]>::from_vec([r])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await?;
        let mut output = HostTensor::<f32, Output>::from_vec(vec![-9876.0; cells])
            .to_hbm::<m![1], Output>(&mut ctx.pdma)
            .await?;
        launch(runtime_rows_norm, (&mut ctx, &count, &input, &mut output)).await?;
        let output = output.to_host::<Output>(&mut ctx.pdma).await?.into_vec();
        let live = usize::try_from(r).unwrap_or(0).min(Rows::SIZE);
        for x in 0..X::SIZE {
            for row in 0..Rows::SIZE {
                let got = output[(x * Rows::SIZE + row) * 8];
                let want = if row < live { expected(row) } else { -9876.0 };
                assert_eq!(got.to_bits(), want.to_bits(), "r={r} slice={x} row={row}");
            }
        }
    }
    Ok(())
}
