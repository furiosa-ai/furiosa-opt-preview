#[cfg(not(backend = "npu"))]
use furiosa_opt_examples::view::padding::view_padding;
use furiosa_opt_examples::view::simpl::view_simpl;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_view_simpl() {
    use furiosa_opt_examples::view::simpl::{A, B};

    let mut ctx = Context::acquire();

    // Create input tensor with shape (A=512)(B=8).
    let input = HostTensor::<i32, m![A, B]>::from_buf((0..4096).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    // Call the device function.
    let output = launch(view_simpl, (&mut *ctx, &input)).await;

    // Verify the output tensor content: [[6,7,0,1,2,3,4,5],[14,15,8,9,10,11,12,13],...].
    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_buf(
            (0..512)
                .flat_map(|x| [6, 7, 0, 1, 2, 3, 4, 5].into_iter().map(move |i| 8 * x + i))
                .collect::<Vec<_>>(),
        )
    );
}

// Why buf-skipped: this test constructs the input via `from_opt_buf` and asserts on `to_buf_opt`
// — both are Simulation/Typecheck-only host-side `Opt<D>` views. Npu / Emulation's host buffer
// is a `Vec<D>` DMA staging area with no `Opt::Uninit` representation, so neither side of the
// assertion has a meaningful equivalent there.
#[cfg(not(backend = "npu"))]
#[tokio::test]
async fn test_view_padding() {
    use furiosa_opt_examples::view::padding::{A, B};

    let mut ctx = Context::acquire();

    // Create input tensor with shape (A=9)(B=7)=63 real values, padded up to 1024 so the DM slice
    // dimension is a valid 64; indices 63..1024 are padding slots.
    let input = HostTensor::<i32, m![[A, B] # 1024]>::from_opt_buf(
        (0..1024)
            .map(|i| if i < 63 { Opt::Init(i) } else { Opt::Uninit })
            .collect::<Vec<_>>(),
    )
    .to_hbm::<m![1], m![[A, B] # 1024]>(&mut ctx.pdma)
    .await;

    // Call the device function.
    let output = launch(view_padding, (&mut *ctx, &input)).await;

    // The view copies rotate each group of 4 as [3,0,1,2] across the whole tensor, with any value
    // >= 63 replaced by the uninitialized padding value.
    assert_eq!(
        output.to_host::<m![[A, B] # 1024]>(&mut ctx.pdma).await.to_buf_opt(),
        Tensor::<i32, m![[A, B] # 1024]>::from_opt_buf(
            (0..256)
                .flat_map(|x| [4 * x + 3, 4 * x, 4 * x + 1, 4 * x + 2])
                .map(|x| if x < 63 { Opt::Init(x) } else { Opt::Uninit })
                .collect::<Vec<_>>(),
        )
        .to_buf_opt(),
    );
}
