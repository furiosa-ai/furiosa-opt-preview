use furiosa_opt_examples::runtime_rows_trf::{Group, Lane, One, Out, Red, runtime_rows_trf};
use furiosa_opt_std::prelude::*;

type Output = m![Group, Out, Lane];

#[tokio::test]
async fn runtime_rows_trf_preserve_tail() -> eyre::Result<()> {
    let mut ctx = Device::new(runtime_rows_trf.topology())?;
    let rows = Group::SIZE * Lane::SIZE;
    let input = HostTensor::<bf16, m![Group, Lane, Red]>::from_vec(
        (0..rows * Red::SIZE).map(|i| bf16::from_f32(f32::from(i16::try_from(i % 17).unwrap()) - 8.0)),
    )
    .to_hbm::<m![1], m![Group, Lane, Red]>(&mut ctx.pdma)
    .await?;
    let weight = HostTensor::<bf16, m![Out, Red]>::from_vec(
        (0..<m![Out, Red]>::SIZE).map(|i| bf16::from_f32(f32::from(i16::try_from(i % 11).unwrap()) - 5.0)),
    )
    .to_hbm::<m![1], m![Out, Red]>(&mut ctx.pdma)
    .await?;
    // Index of row `group * 8 + lane`, output `out` in the Output layout.
    let at = |group: usize, out: usize, lane: usize| (group * Out::SIZE + out) * Lane::SIZE + lane;
    let expected = |group: usize, out: usize, lane: usize| -> f32 {
        let row = group * Lane::SIZE + lane;
        (0..Red::SIZE)
            .map(|k| {
                let a = i16::try_from((row * Red::SIZE + k) % 17).unwrap() - 8;
                let b = i16::try_from((out * Red::SIZE + k) % 11).unwrap() - 5;
                f32::from(a) * f32::from(b)
            })
            .sum()
    };
    for r in [-1, 0, 1, 7, 8, 15, 16, 17] {
        let count = HostTensor::<i32, m![One]>::from_vec([r])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await?;
        let mut output = HostTensor::<f32, Output>::from_vec(vec![-9876.0; rows * Out::SIZE])
            .to_hbm::<m![1], Output>(&mut ctx.pdma)
            .await?;
        launch(runtime_rows_trf, (&mut ctx, &count, &input, &weight, &mut output)).await?;
        let output = output.to_host::<Output>(&mut ctx.pdma).await?.into_vec();
        let live = usize::try_from(r).unwrap_or(0).min(Group::SIZE);
        for group in 0..Group::SIZE {
            for out in 0..Out::SIZE {
                for lane in 0..Lane::SIZE {
                    let value = output[at(group, out, lane)];
                    let want = if group < live {
                        expected(group, out, lane)
                    } else {
                        -9876.0
                    };
                    assert_eq!(
                        value.to_bits(),
                        want.to_bits(),
                        "r={r} group={group} out={out} lane={lane}"
                    );
                }
            }
        }
    }
    Ok(())
}
