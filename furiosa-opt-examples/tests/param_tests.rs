use furiosa_opt_examples::param::{
    A, B, Inputs, Layer, Model, Weight, array_passthrough, borrowed_model_passthrough, explicit_unit,
    local_struct_passthrough, struct_passthrough, tuple_passthrough,
};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn tuple_param_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let out = launch(tuple_passthrough, (&mut *ctx, (&input,))).await;

    assert_eq!(
        out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}

#[tokio::test]
async fn tuple_param_reuses_output() {
    let mut ctx = Context::acquire();
    let first = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let second = first.iter().map(|x| x.wrapping_add(1)).collect::<Vec<_>>();
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first)
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let mut output = HostTensor::<i8, m![A, B]>::zero()
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    launch(tuple_passthrough, (&mut *ctx, (&first_input,)))
        .output(&mut output)
        .await;
    launch(tuple_passthrough, (&mut *ctx, (&second_input,)))
        .output(&mut output)
        .await;

    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(second)
    );
}

#[tokio::test]
async fn struct_param_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let out = launch(struct_passthrough, (&mut *ctx, Inputs { x: &input })).await;

    assert_eq!(
        out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}

#[tokio::test]
async fn array_param_passes() {
    let mut ctx = Context::acquire();
    let first = (0..<m![A, B]>::SIZE).map(|index| index as i8).collect::<Vec<_>>();
    let second = first.iter().map(|value| value.wrapping_add(1)).collect::<Vec<_>>();
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let (first_out, second_out) = launch(
        array_passthrough,
        (&mut *ctx, [Inputs { x: &first_input }, Inputs { x: &second_input }]),
    )
    .await;

    assert_eq!(
        first_out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(first),
    );
    assert_eq!(
        second_out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(second),
    );
}

#[tokio::test]
async fn borrowed_model_param_passes() {
    let mut ctx = Context::acquire();
    let first = (0..<m![A, B]>::SIZE).map(|index| index as i8).collect::<Vec<_>>();
    let second = first.iter().map(|value| value.wrapping_add(1)).collect::<Vec<_>>();
    let first_input = HostTensor::<i8, m![A, B]>::from_vec(first.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let second_input = HostTensor::<i8, m![A, B]>::from_vec(second.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let model = Model {
        layers: [
            Layer {
                weight: Weight { x: &first_input },
            },
            Layer {
                weight: Weight { x: &second_input },
            },
        ],
    };

    let (first_out, second_out) = launch(borrowed_model_passthrough, (&mut *ctx, &model)).await;

    assert_eq!(
        first_out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(first),
    );
    assert_eq!(
        second_out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(second),
    );
}

#[tokio::test]
async fn local_struct_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    let out = launch(local_struct_passthrough, (&mut *ctx, &input)).await;

    assert_eq!(
        out.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data)
    );
}

#[tokio::test]
async fn explicit_unit_passes() {
    let mut ctx = Context::acquire();
    let data = (0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>();
    let input = HostTensor::<i8, m![A, B]>::from_vec(data.clone())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;
    let mut output = HostTensor::<i8, m![A, B]>::zero()
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma)
        .await;

    launch(explicit_unit, (&mut *ctx, &input, &mut output)).await;

    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(data),
    );
}
