//! Rejection fixtures for indirect DMA index placement and noncontiguous table keys.

use furiosa_opt_std::prelude::*;

pub use crate::scatter_gather::{Block, Blocks, D, G, Head, Heads, K, SC, Updates};

axes![SI = 2];

type Chip = m![1];
type PaddedCluster = m![1 # 2];
type BroadcastCluster = m![2];
type PlacedPE = m![G / 128];

/// [`crate::scatter_gather::scatter_pool_rows`] into a pool that keeps each block head-major,
/// `[Blocks, Heads, Block, Head]`, which cannot be compiled: with `Heads` between them, the block
/// axis and the row axis are no longer one run of rows, so the index has no single table axis to
/// count along. Pinned so the rejection keeps naming the two key axes
/// (`relabel_key_axis` in the visa->LIR lowering).
#[device(chip = 1)]
pub fn invalid_scatter_head_major_pool(
    device: &mut Device,
    updates: &HbmTensor<bf16, Chip, m![Updates, Heads, Head]>,
    index: &HbmTensor<i32, Chip, m![Updates]>,
    pool: &mut HbmTensor<bf16, Chip, m![Blocks, Heads, Block, Head]>,
) {
    let updates_dm: DmTensor<bf16, Chip, PaddedCluster, m![Updates, Heads], m![Head]> = updates.to_dm(&mut device.tdma);

    pool.view_mut()
        .scatter::<m![Blocks, Block], m![Heads, Head]>()
        .by_byte_offsets(index.view())
        .from_dm(&mut device.tdma, updates_dm);
}

/// The index spread across the PEs instead of kept in PE 0.
/// A legal `SpmTensor` -- 4 PEs, 512 B each -- but not a legal unscaled index: the gather reads the
/// whole list off PE 0, so a real partition hides three quarters of it.
#[device(chip = 1)]
pub fn invalid_gather_index_across_pes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let index_dm: DmTensor<i32, Chip, BroadcastCluster, m![G / 2], m![G % 2]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, BroadcastCluster, PlacedPE, m![G % 128]> = index_dm.to_spm(&mut device.tdma);
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_positions::<m![G % 128]>(&index_spm)
        .to_dm::<PaddedCluster, m![G / 2], m![G % 2, D]>(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Rejects an index distributed over chips of its own, which names rows the table holds on
/// another chip. There is no inter-chip indirect DMA to fetch them.
#[device(chip = 2)]
pub fn invalid_gather_index_on_other_chips(
    device: &mut Device,
    table: &HbmTensor<bf16, m![SC], m![K, D]>,
    index: &HbmTensor<i32, m![SI], m![G]>,
) -> HbmTensor<bf16, m![SC], m![G, D]> {
    let values: DmTensor<bf16, m![SC], PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}
