use furiosa_opt_examples::slice_span::{
    E, G1, G4, G6, G8, L1, L4, L6, L8, One, R, query_span, query_static_1, query_static_4, query_static_6,
    query_static_8, result_span, result_static_1, result_static_4, result_static_6, result_static_8,
};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

const COUNTS: [i32; 11] = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

fn depth_of(count: i32) -> usize {
    usize::try_from(count).unwrap_or(0).clamp(1, 8)
}

#[tokio::test]
async fn spreads_and_strides_rows_over_runtime_spans() -> eyre::Result<()> {
    let rows = HostTensor::<bf16, m![R, E]>::rand(&mut SmallRng::seed_from_u64(11));
    let rows = rows.into_inner().into_vec();
    let fill = bf16::from_f32(-3.0);
    let mut ctx = Device::new(query_span.topology())?;
    let rows_hbm = HostTensor::<bf16, m![R, E]>::from_vec(rows.clone())
        .to_hbm(&mut ctx.pdma)
        .await?;

    let mut queries = Vec::new();
    let mut results = Vec::new();
    for count in COUNTS {
        let depth = HostTensor::<i32, m![One]>::from_vec([count])
            .to_hbm(&mut ctx.pdma)
            .await?;
        let shift = depth_of(count);
        let query = launch(query_span, (&mut ctx, &depth, &rows_hbm)).await?;
        let query = query.to_host::<m![R, E]>(&mut ctx.pdma).await?.into_inner().into_vec();
        let expected = (0..R::SIZE * E::SIZE).map(|i| rows[((i / E::SIZE) >> shift) * E::SIZE + i % E::SIZE]);
        assert_eq!(query, expected.collect::<Vec<_>>(), "query at {count}");

        let mut out = HostTensor::<bf16, m![R, E]>::from_vec(vec![fill; R::SIZE * E::SIZE])
            .to_hbm(&mut ctx.pdma)
            .await?;
        launch(result_span, (&mut ctx, &depth, &rows_hbm, &mut out)).await?;
        let result = out.to_host::<m![R, E]>(&mut ctx.pdma).await?.into_inner().into_vec();
        let spans = R::SIZE >> shift;
        let expected = (0..R::SIZE * E::SIZE).map(|i| match i / E::SIZE < spans {
            true => rows[((i / E::SIZE) << shift) * E::SIZE + i % E::SIZE],
            false => fill,
        });
        assert_eq!(result, expected.collect::<Vec<_>>(), "result at {count}");
        queries.push((shift, query));
        results.push((shift, result));
    }

    macro_rules! check_static {
        ($query:ident, $result:ident, $depth:literal, $groups:ident, $lanes:ident) => {{
            let mut ctx = Device::new($query.topology())?;
            let firsts = HostTensor::<bf16, m![$groups, E]>::from_vec(rows[..$groups::SIZE * E::SIZE].to_vec())
                .to_hbm(&mut ctx.pdma)
                .await?;
            let query = launch($query, (&mut ctx, &firsts)).await?;
            let query = query
                .to_host::<m![$groups, $lanes, E]>(&mut ctx.pdma)
                .await?
                .into_inner()
                .into_vec();
            let all = HostTensor::<bf16, m![$groups, $lanes, E]>::from_vec(rows.clone())
                .to_hbm(&mut ctx.pdma)
                .await?;
            let mut out = HostTensor::<bf16, m![$groups, E]>::from_vec(vec![fill; $groups::SIZE * E::SIZE])
                .to_hbm(&mut ctx.pdma)
                .await?;
            launch($result, (&mut ctx, &all, &mut out)).await?;
            let result = out
                .to_host::<m![$groups, E]>(&mut ctx.pdma)
                .await?
                .into_inner()
                .into_vec();
            let runtime_query = &queries.iter().find(|run| run.0 == $depth).unwrap().1;
            let runtime_result = &results.iter().find(|run| run.0 == $depth).unwrap().1;
            assert_eq!(runtime_query, &query, "static query at {}", $depth);
            assert_eq!(
                &runtime_result[..$groups::SIZE * E::SIZE],
                &result[..],
                "static result at {}",
                $depth
            );
        }};
    }
    check_static!(query_static_1, result_static_1, 1, G1, L1);
    check_static!(query_static_4, result_static_4, 4, G4, L4);
    check_static!(query_static_6, result_static_6, 6, G6, L6);
    check_static!(query_static_8, result_static_8, 8, G8, L8);
    Ok(())
}
