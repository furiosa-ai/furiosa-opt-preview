//! Rejection fixtures for the two bounds an unscaled gather's SPM index must meet: the per-PE
//! budget its list has to fit, and the PE-0 placement the gather reads it off.

use furiosa_opt_std::prelude::*;

pub use crate::scatter_gather::{D, G, K};

type Chip = m![1];
type PaddedCluster = m![1 # 2];
type BroadcastCluster = m![2];
type PlacedPE = m![G / 128];

/// The index spread across the PEs instead of kept in PE 0.
/// A legal `SpmTensor` -- 4 PEs, 512 B each -- but not a legal unscaled index: the gather reads the
/// whole list off PE 0, so a real partition hides three quarters of it.
#[device(chip = 1)]
pub fn invalid_gather_unscaled_index_across_pes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let index_dm: DmTensor<i32, Chip, BroadcastCluster, m![G / 2], m![G % 2]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, BroadcastCluster, PlacedPE, m![G % 128]> = index_dm.to_spm(&mut device.tdma);
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, D]> = table.dma_gather_unscaled(&index_spm);

    values_dm.to_hbm(&mut device.tdma)
}
