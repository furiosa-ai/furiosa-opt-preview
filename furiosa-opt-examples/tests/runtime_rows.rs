use furiosa_opt_examples::runtime_rows::{One, Out, Red, Rows, runtime_rows};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn runtime_rows_preserve_tail() -> eyre::Result<()> {
    let mut ctx = Device::new(runtime_rows.topology())?;
    let input = HostTensor::<bf16, m![Rows, Red]>::from_vec(
        (0..<m![Rows, Red]>::SIZE).map(|i| bf16::from_f32(f32::from(i16::try_from(i % 17).unwrap()) - 8.0)),
    )
    .to_hbm::<m![1], m![Rows, Red]>(&mut ctx.pdma)
    .await?;
    let weight = HostTensor::<bf16, m![Out, Red]>::from_vec(
        (0..<m![Out, Red]>::SIZE).map(|i| bf16::from_f32(f32::from(i16::try_from(i % 11).unwrap()) - 5.0)),
    )
    .to_hbm::<m![1], m![Out, Red]>(&mut ctx.pdma)
    .await?;
    let mut full = HostTensor::<f32, m![Rows, Out]>::from_vec(vec![-9876.0; Rows::SIZE * Out::SIZE])
        .to_hbm::<m![1], m![Rows, Out]>(&mut ctx.pdma)
        .await?;
    let count = HostTensor::<i32, m![One]>::from_vec([128])
        .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
        .await?;
    launch(runtime_rows, (&mut ctx, &count, &input, &weight, &mut full)).await?;
    let full = full.to_host::<m![Rows, Out]>(&mut ctx.pdma).await?.into_vec();
    for (index, value) in full.iter().enumerate() {
        let expected: f32 = (0..Red::SIZE)
            .map(|k| {
                let a = i16::try_from((index / Out::SIZE * Red::SIZE + k) % 17).unwrap() - 8;
                let b = i16::try_from((index % Out::SIZE * Red::SIZE + k) % 11).unwrap() - 5;
                f32::from(a) * f32::from(b)
            })
            .sum();
        assert_eq!(value.to_bits(), expected.to_bits(), "full result at index={index}");
    }

    for r in [-1, 0, 1, 7, 8, 64, 127, 128, 129] {
        let count = HostTensor::<i32, m![One]>::from_vec([r])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await?;
        let mut output = HostTensor::<f32, m![Rows, Out]>::from_vec(vec![-9876.0; Rows::SIZE * Out::SIZE])
            .to_hbm::<m![1], m![Rows, Out]>(&mut ctx.pdma)
            .await?;
        launch(runtime_rows, (&mut ctx, &count, &input, &weight, &mut output)).await?;
        let output = output.to_host::<m![Rows, Out]>(&mut ctx.pdma).await?.into_vec();
        let live = usize::try_from(r).unwrap_or(0).min(Rows::SIZE) * Out::SIZE;
        assert_eq!(
            output[..live].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            full[..live].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "live prefix at r={r}",
        );
        assert!(
            output[live..]
                .iter()
                .all(|value| value.to_bits() == (-9876.0f32).to_bits())
        );
    }
    Ok(())
}
