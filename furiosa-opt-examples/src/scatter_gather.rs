//! Scatter/gather test kernels.
//!
//! `scatter_minimal` scatters `K = 512` keys into a `C = 612` cache (sparse: `C > K`);
//! the non-power-of-2, larger-than-`K` cache stresses non-aligned scatter coverage.
//!
//! `gather_minimal` gathers `G = 512` rows from the `K`-row table. Its gather count `G` is
//! separate from the scatter cache `C` so the values partition `G / 2 = 256` equals
//! `num_slices` (the sparse `C / 2 = 306` would round up past it and fail to lower).
//! `gather_unaligned` covers a non-power-of-2 count (`U = 768`, 3 rows/slice) that still meets
//! the 256-slice constraint.

use furiosa_opt_std::prelude::*;

axes![
    K = 512, // Scatter key
    D = 128, // Payload per key
    C = 612, // Cache length (non-power-of-2, > K: stresses unaligned coverage)
    G = 512, // Slice-aligned gather count (G / 2 = 256)
    U = 768  // Unaligned gather count (non-power-of-2, > K: 3 rows/slice, U / 3 = 256)
];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Scatter values into cache at index positions.
#[device(chip = 1)]
pub fn scatter_minimal(
    ctx: &mut Context,
    data: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![K]>,
    output: &mut HbmTensor<bf16, Chip, m![C, D]>,
) {
    let data_dm: DmTensor<bf16, Chip, Cluster, m![K / 2], m![K % 2, D]> = data.to_dm(&mut ctx.tdma);

    data_dm.dma_scatter::<m![K], _, _>(index, output);
}

/// Gather `G` rows from `table` at positions given by `index` into a fresh HBM output.
///
/// Inverse of [`scatter_minimal`]. Produces a `DmTensor` via [`HbmTensor::dma_gather`], then
/// writes it back to HBM via [`DmTensor::to_hbm`], returned by value because `dma_gather` does
/// not write into an existing `&mut HbmTensor`. The `G / 2 = 256` values partition equals
/// `num_slices`, so it lowers end to end through VISA -> LIR -> EDF.
#[device(chip = 1)]
pub fn gather_minimal(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let values_dm: DmTensor<bf16, Chip, Cluster, m![G / 2], m![G % 2, D]> = table.dma_gather(index);

    values_dm.to_hbm(&mut ctx.tdma, 0x1000)
}

/// Gather an unaligned count `U = 768` (`= 3 * num_slices`, non-power-of-2, `> K`) from the
/// `K`-row table. Unlike [`gather_minimal`]'s `2` rows/slice, the values partition here packs
/// `3` rows/slice (`U / 3 = 256 = num_slices`), so it still lowers to EDF while covering a
/// non-power-of-2 gather size that wraps the table.
#[device(chip = 1)]
pub fn gather_unaligned(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![U]>,
) -> HbmTensor<bf16, Chip, m![U, D]> {
    let values_dm: DmTensor<bf16, Chip, Cluster, m![U / 3], m![U % 3, D]> = table.dma_gather(index);

    values_dm.to_hbm(&mut ctx.tdma, 0x1000)
}
