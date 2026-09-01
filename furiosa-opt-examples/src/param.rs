//! Tuple-, struct-, array-, and borrowed-typed `#[device]` parameter examples.
//!
//! Each passthrough returns its input tensor unchanged (HBM -> SRAM -> HBM), so the
//! tests can confirm every parameter shape reaches the kernel by checking
//! the data survives the round-trip.

use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

#[derive(DeviceSend)]
pub struct Inputs<'a> {
    pub x: &'a HbmTensor<i8, Chip, m![A, B]>,
}

#[derive(DeviceSend)]
pub struct Weight<'a> {
    pub x: &'a HbmTensor<i8, Chip, m![A, B]>,
}

#[derive(DeviceSend)]
pub struct Layer<'a> {
    pub weight: Weight<'a>,
}

#[derive(DeviceSend)]
pub struct Model<'a, const N: usize> {
    pub layers: [Layer<'a>; N],
}

fn copy(ctx: &mut Context, input: &HbmTensor<i8, Chip, m![A, B]>) -> HbmTensor<i8, Chip, m![A, B]> {
    input
        .to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma)
        .to_hbm(&mut ctx.tdma)
}

#[device(chip = 1)]
pub fn tuple_passthrough(
    ctx: &mut Context,
    inputs: (&HbmTensor<i8, Chip, m![A, B]>,),
) -> HbmTensor<i8, Chip, m![A, B]> {
    copy(ctx, inputs.0)
}

#[device(chip = 1)]
pub fn struct_passthrough(ctx: &mut Context, inputs: Inputs<'_>) -> HbmTensor<i8, Chip, m![A, B]> {
    copy(ctx, inputs.x)
}

#[device(chip = 1)]
pub fn array_passthrough(
    ctx: &mut Context,
    inputs: [Inputs<'_>; 2],
) -> (HbmTensor<i8, Chip, m![A, B]>, HbmTensor<i8, Chip, m![A, B]>) {
    let [first, second] = inputs;
    (copy(ctx, first.x), copy(ctx, second.x))
}

#[device(chip = 1)]
pub fn borrowed_model_passthrough(
    ctx: &mut Context,
    model: &Model<'_, 2>,
) -> (HbmTensor<i8, Chip, m![A, B]>, HbmTensor<i8, Chip, m![A, B]>) {
    (copy(ctx, model.layers[0].weight.x), copy(ctx, model.layers[1].weight.x))
}

#[device(chip = 1)]
pub fn local_struct_passthrough(ctx: &mut Context, x: &HbmTensor<i8, Chip, m![A, B]>) -> HbmTensor<i8, Chip, m![A, B]> {
    let inputs = Inputs { x };
    copy(ctx, inputs.x)
}

#[device(chip = 1)]
pub fn explicit_unit(
    ctx: &mut Context,
    input: &HbmTensor<i8, Chip, m![A, B]>,
    output: &mut HbmTensor<i8, Chip, m![A, B]>,
) -> () {
    input
        .to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma)
        .view()
        .to_hbm_view(&mut ctx.tdma, output.view_mut());
}
