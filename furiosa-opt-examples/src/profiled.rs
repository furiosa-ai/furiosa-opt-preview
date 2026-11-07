//! Kernels whose profile spans have an answer known from the hardware's order rules: two
//! contractions back to back on one context, two loads queued on one DMA queue, and a loop of row
//! transfers to select trips of.

use furiosa_opt_std::prelude::*;

axes![PE = 8, LOCAL = 64, E2K = 2048];
axes![AX = 4096, K8 = 32, R = 8];
axes![TA = 512, TB = 32];

type Chip = m![1];
type Cluster = m![PE / 4];
type Slice = m![PE % 4, LOCAL];
type One = m![1 # 2];
type Single = m![1 # 256];
type Lane = m![R];

/// 4 MiB, a slice's share 8 KiB.
pub type Bulk = HbmTensor<i32, Chip, m![PE, LOCAL, E2K]>;
pub type Rows = HbmTensor<i8, Chip, m![AX, K8]>;
pub type Weight = HbmTensor<i8, Chip, m![R, K8]>;
pub type Products = HbmTensor<i32, Chip, m![AX, R]>;

fn contract(
    device: &mut Device,
    input: &DmTensor<i8, Chip, One, Single, m![AX, K8]>,
    trf: &TrfTensor<i8, Chip, One, Single, Lane, m![K8]>,
) -> DmTensor<i32, Chip, One, Single, m![AX, R]> {
    device
        .main
        .begin(input.view())
        .fetch::<m![AX], m![K8]>()
        .fetch_cast::<i8>()
        .collect::<m![AX], m![K8]>()
        .contract_outer::<m![AX], m![K8], _, _, _>(trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![AX]>()
        .contract_lane::<m![AX], m![R]>(LaneMode::Interleaved)
        .commit_trim::<m![R]>()
        .commit()
}

/// The same contraction twice on the main context, the second issued while the first runs.
#[device(chip = 1)]
pub fn contract_pair(device: &mut Device, input: &Rows, weight: &Weight) -> (Products, Products) {
    let input = input.to_dm::<One, Single, m![AX, K8]>(&mut device.tdma);
    let weight = weight.to_dm::<One, Single, m![R, K8]>(&mut device.tdma);
    let trf: TrfTensor<i8, Chip, One, Single, Lane, m![K8]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![R], m![K8]>()
        .fetch_cast::<i8>()
        .collect::<m![R], m![K8]>()
        .to_trf();
    let first = contract(device, &input, &trf);
    let second = contract(device, &input, &trf);
    (first.to_hbm(&mut device.tdma), second.to_hbm(&mut device.tdma))
}

/// Two 4 MiB loads issued back to back, so the second waits in the queue behind the first.
#[device(chip = 1)]
pub fn load_pair(device: &mut Device, first: &Bulk, second: &Bulk) -> (Bulk, Bulk) {
    let first = first.to_dm::<Cluster, Slice, m![E2K]>(&mut device.tdma);
    let second = second.to_dm::<Cluster, Slice, m![E2K]>(&mut device.tdma);
    let (mut left, mut right): (Bulk, Bulk) = (HbmTensor::new(), HbmTensor::new());
    first.view().to_hbm_view(&mut device.tdma, left.view_mut());
    second.view().to_hbm_view(&mut device.tdma, right.view_mut());
    (left, right)
}

fn transpose_row(
    device: &mut Device,
    input: &HbmTensorView<'_, i8, Chip, m![TA, TB]>,
    output: &mut HbmTensor<i8, Chip, m![TB, TA]>,
    row: usize,
) {
    let source = input.tile::<m![TB], 1, m![TA, 1 # 32]>(row);
    source.to_hbm_view(
        &mut device.tdma,
        output.view_mut().tile::<m![TB], 1, m![1 #{!} 32, TA]>(row),
    );
}

/// A rolled loop of 32 trips, each transposing one row of `input` into two outputs.
#[device(chip = 1)]
pub fn rows(
    device: &mut Device,
    input: HbmTensorView<'_, i8, Chip, m![TA, TB]>,
) -> (HbmTensor<i8, Chip, m![TB, TA]>, HbmTensor<i8, Chip, m![TB, TA]>) {
    let mut first = HbmTensor::<i8, Chip, m![TB, TA]>::new();
    let mut second = HbmTensor::<i8, Chip, m![TB, TA]>::new();
    for row in 0..32 {
        transpose_row(device, &input, &mut first, row);
        transpose_row(device, &input, &mut second, row);
    }
    (first, second)
}

axes![LA = 64, LB = 2048];

/// A rolled loop of 2048 one-row transfers: its every command recorded overruns a chunk's queue.
#[device(chip = 1)]
pub fn many_rows(
    device: &mut Device,
    input: HbmTensorView<'_, i8, Chip, m![LA, LB]>,
) -> HbmTensor<i8, Chip, m![LB, LA]> {
    let mut output = HbmTensor::<i8, Chip, m![LB, LA]>::new();
    for row in 0..2048 {
        let source = input.tile::<m![LB], 1, m![LA, 1 # 2048]>(row);
        source.to_hbm_view(
            &mut device.tdma,
            output.view_mut().tile::<m![LB], 1, m![1 #{!} 2048, LA]>(row),
        );
    }
    output
}
