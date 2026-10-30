use furiosa_opt_examples::runtime_rows_vector::{H, One, Rows, runtime_rows_add};
use furiosa_opt_std::prelude::*;

const SENTINEL: f32 = -9876.0;

fn operand(seed: usize) -> Vec<f32> {
    (0..Rows::SIZE * H::SIZE)
        .map(|i| f32::from(u16::try_from((i * seed) % 1009).unwrap()) * 0.25)
        .collect()
}

#[tokio::test]
async fn vector_rows_preserve_tail() -> eyre::Result<()> {
    let mut ctx = Device::new(runtime_rows_add.topology())?;
    let (lhs, rhs) = (operand(7), operand(13));
    let sum: Vec<f32> = lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect();
    let lhs = HostTensor::<f32, m![Rows, H]>::from_vec(lhs)
        .to_hbm::<m![1], m![Rows, H]>(&mut ctx.pdma)
        .await?;
    let rhs = HostTensor::<f32, m![Rows, H]>::from_vec(rhs)
        .to_hbm::<m![1], m![Rows, H]>(&mut ctx.pdma)
        .await?;
    for r in [-1, 0, 1, 7, 8, 64, 127, 128, 129] {
        let count = HostTensor::<i32, m![One]>::from_vec([r])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await?;
        let mut output = HostTensor::<f32, m![Rows, H]>::from_vec(vec![SENTINEL; Rows::SIZE * H::SIZE])
            .to_hbm::<m![1], m![Rows, H]>(&mut ctx.pdma)
            .await?;
        launch(runtime_rows_add, (&mut ctx, &count, &lhs, &rhs, &mut output)).await?;
        let output = output.to_host::<m![Rows, H]>(&mut ctx.pdma).await?.into_vec();
        let live = usize::try_from(r).unwrap_or(0).min(Rows::SIZE) * H::SIZE;
        let bits = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&output[..live]), bits(&sum[..live]), "live prefix at r={r}");
        assert!(
            output[live..].iter().all(|x| x.to_bits() == SENTINEL.to_bits()),
            "suffix at r={r}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn grouped_rows_cover_whole_groups() -> eyre::Result<()> {
    use furiosa_opt_examples::runtime_rows_vector_silicon::grouped::{self, H, One, Rows};
    let mut ctx = Device::new(grouped::rows.topology())?;
    let values = |seed: usize| {
        (0..Rows::SIZE * H::SIZE).map(move |i| f32::from(u16::try_from((i * seed + i / H::SIZE) % 16).unwrap()))
    };
    let sum: Vec<u16> = values(7)
        .zip(values(13))
        .map(|(a, b)| bf16::from_f32(a + b).to_bits())
        .collect();
    let lhs = HostTensor::<bf16, m![Rows, H]>::from_vec(values(7).map(bf16::from_f32))
        .to_hbm::<m![1], m![Rows, H]>(&mut ctx.pdma)
        .await?;
    let rhs = HostTensor::<bf16, m![Rows, H]>::from_vec(values(13).map(bf16::from_f32))
        .to_hbm::<m![1], m![Rows, H]>(&mut ctx.pdma)
        .await?;
    let sentinel = bf16::from_f32(SENTINEL);
    for r in [-1, 0, 1, 7, 8, 9, 64, 127, 128, 129] {
        let count = HostTensor::<i32, m![One]>::from_vec([r])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await?;
        let mut output = HostTensor::<bf16, m![Rows, H]>::from_vec(vec![sentinel; Rows::SIZE * H::SIZE])
            .to_hbm::<m![1], m![Rows, H]>(&mut ctx.pdma)
            .await?;
        launch(grouped::rows, (&mut ctx, &count, &lhs, &rhs, &mut output)).await?;
        let output: Vec<u16> = output
            .to_host::<m![Rows, H]>(&mut ctx.pdma)
            .await?
            .into_vec()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let live = (usize::try_from(r).unwrap_or(0).div_ceil(8) * 8).min(Rows::SIZE) * H::SIZE;
        assert_eq!(output[..live], sum[..live], "live groups at r={r}");
        assert!(
            output[live..].iter().all(|x| *x == sentinel.to_bits()),
            "rows past the groups at r={r}"
        );
    }
    Ok(())
}
