use furiosa_opt_examples::reduce_span::{
    D, G, G5, G6, G7, G8, H, L5, L6, L7, L8, One, R, max_span, wave, wave_static_5, wave_static_6, wave_static_7,
    wave_static_8,
};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Every slice of an aligned block of `2^depth` slices holds the block's fold, from `identity`.
fn span_oracle(input: &[f32], cells: usize, depth: usize, identity: f32, f: fn(f32, f32) -> f32) -> Vec<f32> {
    let span = 1 << depth.clamp(1, 8);
    (0..R::SIZE * cells)
        .map(|index| {
            let (slice, cell) = (index / cells, index % cells);
            let first = slice & !(span - 1);
            (first..first + span).fold(identity, |acc, source| f(acc, input[source * cells + cell]))
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Expands a static result, one row per group, to the slices of each group.
fn per_slice(grouped: &[f32], cells: usize, depth: usize) -> Vec<f32> {
    (0..R::SIZE * cells)
        .map(|index| grouped[((index / cells) >> depth) * cells + index % cells])
        .collect()
}

#[tokio::test]
async fn reduces_each_aligned_group_of_two_to_the_depth() -> eyre::Result<()> {
    let mut rng = SmallRng::seed_from_u64(7);
    let scores = HostTensor::<f32, m![R, H, G]>::rand(&mut rng);
    let values = HostTensor::<f32, m![R, H, G, D]>::rand(&mut rng);
    let (score_cells, value_cells) = (H::SIZE * G::SIZE, H::SIZE * G::SIZE * D::SIZE);
    let mut ctx = Device::new(wave.topology())?;
    let scores_hbm = scores.to_hbm(&mut ctx.pdma).await?;
    let values_hbm = values.to_hbm(&mut ctx.pdma).await?;
    let (scores, values) = (scores.into_inner().into_vec(), values.into_inner().into_vec());

    let mut runs = Vec::new();
    for count in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
        let depth = HostTensor::<i32, m![One]>::from_vec([count])
            .to_hbm(&mut ctx.pdma)
            .await?;
        let (max, sum, pv) = launch(wave, (&mut ctx, &depth, &scores_hbm, &values_hbm)).await?;
        let max = max.to_host::<m![R, H, G]>(&mut ctx.pdma).await?.into_inner().into_vec();
        let sum = sum.to_host::<m![R, H, G]>(&mut ctx.pdma).await?.into_inner().into_vec();
        let pv = pv
            .to_host::<m![R, H, G, D]>(&mut ctx.pdma)
            .await?
            .into_inner()
            .into_vec();
        let depth = usize::try_from(count).unwrap_or(0).clamp(1, 8);
        assert_eq!(
            bits(&max),
            bits(&span_oracle(&scores, score_cells, depth, f32::MIN, f32::max)),
            "max at {count}"
        );
        assert_eq!(
            bits(&sum),
            bits(&span_oracle(&scores, score_cells, depth, 0.0, |a, b| a + b)),
            "sum at {count}"
        );
        assert_eq!(
            bits(&pv),
            bits(&span_oracle(&values, value_cells, depth, 0.0, |a, b| a + b)),
            "pv at {count}"
        );
        runs.push((count, max, sum, pv));
    }

    macro_rules! check_static {
        ($kernel:ident, $depth:literal, $groups:ident, $lanes:ident) => {{
            let mut ctx = Device::new($kernel.topology())?;
            let scores_hbm = HostTensor::<f32, m![$groups, $lanes, H, G]>::from_vec(scores.clone())
                .to_hbm(&mut ctx.pdma)
                .await?;
            let values_hbm = HostTensor::<f32, m![$groups, $lanes, H, G, D]>::from_vec(values.clone())
                .to_hbm(&mut ctx.pdma)
                .await?;
            let (max, sum, pv) = launch($kernel, (&mut ctx, &scores_hbm, &values_hbm)).await?;
            let max = max
                .to_host::<m![$groups, H, G]>(&mut ctx.pdma)
                .await?
                .into_inner()
                .into_vec();
            let sum = sum
                .to_host::<m![$groups, H, G]>(&mut ctx.pdma)
                .await?
                .into_inner()
                .into_vec();
            let pv = pv
                .to_host::<m![$groups, H, G, D]>(&mut ctx.pdma)
                .await?
                .into_inner()
                .into_vec();
            let (_, runtime_max, runtime_sum, runtime_pv) = runs.iter().find(|run| run.0 == $depth).unwrap();
            assert_eq!(
                bits(runtime_max),
                bits(&per_slice(&max, score_cells, $depth)),
                "static max at {}",
                $depth
            );
            assert_eq!(
                bits(runtime_sum),
                bits(&per_slice(&sum, score_cells, $depth)),
                "static sum at {}",
                $depth
            );
            assert_eq!(
                bits(runtime_pv),
                bits(&per_slice(&pv, value_cells, $depth)),
                "static pv at {}",
                $depth
            );
        }};
    }
    check_static!(wave_static_5, 5, G5, L5);
    check_static!(wave_static_6, 6, G6, L6);
    check_static!(wave_static_7, 7, G7, L7);
    check_static!(wave_static_8, 8, G8, L8);
    Ok(())
}

#[tokio::test]
async fn max_span_matches_the_full_reduce_at_the_capacity() -> eyre::Result<()> {
    let mut ctx = Device::new(max_span.topology())?;
    let scores = HostTensor::<f32, m![R, H, G]>::rand(&mut SmallRng::seed_from_u64(3));
    let scores_hbm = scores.to_hbm(&mut ctx.pdma).await?;
    let depth = HostTensor::<i32, m![One]>::from_vec([8]).to_hbm(&mut ctx.pdma).await?;
    let max = launch(max_span, (&mut ctx, &depth, &scores_hbm)).await?;
    let max = max.to_host::<m![R, H, G]>(&mut ctx.pdma).await?.into_inner().into_vec();
    let cells = H::SIZE * G::SIZE;
    let expected = span_oracle(&scores.into_inner().into_vec(), cells, 8, f32::MIN, f32::max);
    assert_eq!(bits(&max), bits(&expected));
    Ok(())
}
