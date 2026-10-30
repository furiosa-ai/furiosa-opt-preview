//! Local sum types preserve the selected tensor layout through helper calls.
//! Selection here is compile-time. Enums do not introduce a device-entry ABI.
use furiosa_opt_std::prelude::*;

axes![Rows = 512, Cols = 32];
type Hbm = HbmTensor<i32, m![1], m![Rows, Cols]>;
type RowMajor = DmTensor<i32, m![1], m![Rows / 256], m![Rows % 256], m![Cols]>;
type ColumnMajor = DmTensor<i32, m![1], m![Rows / 256], m![Rows % 256 / 8, Cols / 4], m![Rows % 8, Cols % 4]>;

#[repr(u8)]
enum Layout {
    Rows = 3,
    Columns = 19,
}
enum Tensor {
    Rows(RowMajor),
    Columns { value: ColumnMajor },
}

trait Module<I> {
    type Output;
    fn forward(&self, input: I) -> Self::Output;
}
struct Identity;
impl<I> Module<I> for Identity {
    type Output = I;
    fn forward(&self, input: I) -> I {
        input
    }
}

fn stage<const COLUMNS: bool>(device: &mut Device, input: &Hbm) -> Tensor {
    match const { if COLUMNS { Layout::Columns } else { Layout::Rows } } {
        Layout::Rows => Tensor::Rows(input.to_dm(&mut device.tdma)),
        Layout::Columns => Tensor::Columns {
            value: input.to_dm(&mut device.tdma),
        },
    }
}

fn store(device: &mut Device, tensor: &Tensor) -> Hbm {
    match tensor {
        Tensor::Rows(value) => value.to_hbm(&mut device.tdma),
        Tensor::Columns { value } => value.to_hbm(&mut device.tdma),
    }
}

#[device(chip = 1)]
pub fn rows(device: &mut Device, input: &Hbm) -> Hbm {
    let selected = stage::<false>(device, input);
    store(device, Identity.forward(&selected))
}

#[device(chip = 1)]
pub fn columns(device: &mut Device, input: &Hbm) -> Hbm {
    let selected = stage::<true>(device, input);
    store(device, Identity.forward(&selected))
}

fn checked(device: &mut Device, input: &Hbm) -> Result<Tensor, ()> {
    let selected = Identity.forward(Ok(stage::<false>(device, input)))?;
    Ok(selected)
}

#[device(chip = 1)]
pub fn result(device: &mut Device, input: &Hbm, output: &mut Hbm) {
    if let Ok(Tensor::Rows(selected)) = checked(device, input) {
        selected.view().to_hbm_view(&mut device.tdma, output.view_mut());
    }
}

axes![Entry = 4, Key = 2];
pub type Block = HbmTensor<i32, m![1], m![Entry]>;

/// Layer state selected inside an unrolled model loop retains its allocation identity.
#[device(chip = 1)]
pub fn update_layers(
    device: &mut Device,
    layers: &mut [Slice<Block>; 2],
    indices: &HbmTensor<i32, m![1], m![Key]>,
    values: &HbmTensor<i32, m![1], m![Key, Entry]>,
) {
    let values: DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Key, Entry]> = values.to_dm(&mut device.tdma);
    #[allow(clippy::needless_range_loop, reason = "Exercises unrolled mutable array indexing")]
    #[unroll]
    for layer in 0..2 {
        layers[layer]
            .scatter::<m![1], m![1 # 2, 1 # 256, Entry]>()
            .by_byte_offsets(indices.view())
            .from_dm_view(&mut device.tdma, values.view());
    }
}

#[device(chip = 1)]
pub fn update_blocks(
    device: &mut Device,
    layers: &mut [Slice<Block>; 2],
    indices: &HbmTensor<i32, m![1], m![Key]>,
    values: &HbmTensor<i32, m![1], m![Key]>,
) {
    let values: DmTensor<i32, m![1], m![1 # 2], m![1 # 256], m![Key]> = values.to_dm(&mut device.tdma);
    #[allow(clippy::needless_range_loop, reason = "Exercises unrolled mutable array indexing")]
    #[unroll]
    for layer in 0..2 {
        layers[layer]
            .scatter::<m![Entry], m![1]>()
            .by_byte_offsets(indices.view())
            .from_dm_view(&mut device.tdma, values.view());
    }
}

fn identity<T>(value: T) -> T {
    value
}

/// Captured DMA access survives nested maps, including a unit-valued inner map.
#[device(chip = 1)]
pub fn map_tensors(device: &mut Device, first: &Hbm, second: &Hbm, first_output: &mut Hbm, second_output: &mut Hbm) {
    let inputs = [first, second].map(identity);
    let values: [RowMajor; 2] = inputs.map(|input| input.to_dm(&mut device.tdma));
    let [first, second] = &values;
    [(first, first_output), (second, second_output)].map(|(value, output)| {
        [value, value].map(|value| {
            value.view().to_hbm_view(&mut device.tdma, output.view_mut());
        })
    });
}

/// A closure called with the device and its output as arguments issues its copy on that device.
#[device(chip = 1)]
pub fn pass_device(device: &mut Device, input: &Hbm, first: &mut Hbm, second: &mut Hbm) {
    let copy = |device: &mut Device, output: &mut Hbm| input.view().to_hbm_view(&mut device.tdma, output.view_mut());
    copy(device, first);
    copy(device, second);
}

fn lend(device: &mut Device, run: impl FnOnce(&mut Device)) {
    run(device)
}

/// A closure a generic caller runs with the device issues its copy on that device.
#[device(chip = 1)]
pub fn lend_device(device: &mut Device, input: &Hbm, output: &mut Hbm) {
    lend(device, |device| {
        input.view().to_hbm_view(&mut device.tdma, output.view_mut())
    });
}

/// As [`lend_device`], with an `FnMut` closure the caller runs through `FnOnce`.
#[device(chip = 1)]
pub fn lend_stored(device: &mut Device, input: &Hbm, output: &mut Hbm) {
    let copy = |device: &mut Device| input.view().to_hbm_view(&mut device.tdma, output.view_mut());
    lend(device, copy);
}
