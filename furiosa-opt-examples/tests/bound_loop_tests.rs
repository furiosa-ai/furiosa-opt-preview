use furiosa_opt_examples::bound_loop::{
    Slice, Width, increment_to_tile_boundary, loop_with_runtime_end, nested_dynamic_loops, runtime_end_reused_in_body,
};
use furiosa_opt_std::prelude::*;

fn input_host() -> HostTensor<i32, m![Slice, Width]> {
    HostTensor::from_vec((0..<m![Slice, Width]>::SIZE).map(|i| i as i32 % 23 - 11))
}

/// The input with `iterations` added to every element.
fn expected(iterations: i32) -> Vec<i32> {
    input_host().into_inner().map(|x| x + iterations).into_vec()
}

/// The loop runs until its runtime end, and a negative `i32` end runs it zero times:
/// `output == input + count.max(0)`.
#[tokio::test]
async fn test_loop_with_runtime_end() -> eyre::Result<()> {
    let mut device = Device::new(loop_with_runtime_end.topology())?;
    let input_hbm = input_host().to_hbm(&mut device.pdma).await?;
    for (count, iterations) in [(-1, 0), (0, 0), (4, 4)] {
        let count_hbm = HbmScalar::<i32>::from_host(count, &mut device.pdma).await?;
        let output = launch(loop_with_runtime_end, (&mut device, &input_hbm, &count_hbm)).await?;
        assert_eq!(
            expected(iterations),
            output.to_host::<m![Slice, Width]>(&mut device.pdma).await?.into_vec(),
            "count={count}"
        );
    }
    Ok(())
}

/// Explicit `(rows, offset, iterations)` triples check the count without repeating its arithmetic:
/// an empty span, a span cut short by the live rows, a full span, an aligned offset, and the
/// bound 7.
#[tokio::test]
async fn test_increment_to_tile_boundary() -> eyre::Result<()> {
    let mut device = Device::new(increment_to_tile_boundary.topology())?;
    let input_hbm = input_host().to_hbm(&mut device.pdma).await?;

    for (rows, offset, iterations) in [(1, 3, 0), (4, 3, 1), (32, 3, 5), (32, 0, 0), (32, 1, 7)] {
        let rows_hbm = HbmScalar::<i32>::from_host(rows, &mut device.pdma).await?;
        let offset_hbm = HbmScalar::<i32>::from_host(offset, &mut device.pdma).await?;
        let output = launch(
            increment_to_tile_boundary,
            (&mut device, &input_hbm, &rows_hbm, &offset_hbm),
        )
        .await?;
        assert_eq!(
            expected(iterations),
            output.to_host::<m![Slice, Width]>(&mut device.pdma).await?.into_vec(),
            "rows={rows}, offset={offset}"
        );
    }
    Ok(())
}

/// Outer iteration `i` runs its inner loop `i` times, so `count * (count - 1) / 2` in total; the
/// first iteration runs it zero times.
#[tokio::test]
async fn test_nested_dynamic_loops() -> eyre::Result<()> {
    let mut device = Device::new(nested_dynamic_loops.topology())?;
    let input_hbm = input_host().to_hbm(&mut device.pdma).await?;
    for (count, iterations) in [(0, 0), (1, 0), (4, 6)] {
        let count_hbm = HbmScalar::<i32>::from_host(count, &mut device.pdma).await?;
        let output = launch(nested_dynamic_loops, (&mut device, &input_hbm, &count_hbm)).await?;
        assert_eq!(
            expected(iterations),
            output.to_host::<m![Slice, Width]>(&mut device.pdma).await?.into_vec(),
            "count={count}"
        );
    }
    Ok(())
}

/// The outer and inner loops share one runtime end, so the body runs `count * count` times.
#[tokio::test]
async fn test_runtime_end_reused_in_body() -> eyre::Result<()> {
    let mut device = Device::new(runtime_end_reused_in_body.topology())?;
    let input_hbm = input_host().to_hbm(&mut device.pdma).await?;
    for (count, iterations) in [(0, 0), (2, 4), (4, 16)] {
        let count_hbm = HbmScalar::<i32>::from_host(count, &mut device.pdma).await?;
        let output = launch(runtime_end_reused_in_body, (&mut device, &input_hbm, &count_hbm)).await?;
        assert_eq!(
            expected(iterations),
            output.to_host::<m![Slice, Width]>(&mut device.pdma).await?.into_vec(),
            "count={count}"
        );
    }
    Ok(())
}
