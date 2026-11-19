use furiosa_opt_std::prelude::*;

axes![C = 2, P = 4, E = 16, LengthClusters = 2, LengthPes = 4];
type Chip = m![1];
type Cluster = m![2];
type Pe0 = m![1 # 4];
pub type Vector = HbmTensor<i32, Chip, m![E]>;
pub type Distributed = HbmTensor<i32, Chip, m![P, E]>;
pub type Clustered = HbmTensor<i32, Chip, m![C, P, E]>;

/// Exercises scalar operators and index widths.
#[device(chip = 1)]
pub fn scalar_binary_operators(device: &mut Device, input: &LocalLengths) -> LocalLengths {
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = input.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        let a = values.get(0u32);
        let b = values.get(1i64);
        values.set(0u64, a.min(b));
        values.set(1usize, a.max(b));
        values.set(2i32, a + b);
        values.set(3u32, (a * b) as i64 as i32);
    });
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn dma_clustered(device: &mut Device, input: &Clustered) -> Clustered {
    let values: SpmTensor<i32, Chip, m![C], m![P], m![E]> = input.to_spm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn dma_clustered_redistribute(device: &mut Device, input: &Clustered) -> Clustered {
    let values: SpmTensor<i32, Chip, m![C], m![P], m![E]> = input.to_spm(&mut device.tdma);
    let values: SpmTensor<i32, Chip, m![C], Pe0, m![P, E]> = values.to_spm(&mut device.tdma);
    let values: SpmTensor<i32, Chip, m![C], m![P], m![E]> = values.to_spm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn initialize(device: &mut Device, scale: &HbmScalar<i32>) -> Vector {
    let scale = scale.to_spm(&mut device.tdma);
    let mut values = SpmTensor::<i32, Chip, Cluster, Pe0, m![E]>::zero();
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, scale + i as i32);
        }
    });
    values.to_hbm(&mut device.tdma)
}

/// Subtracts a replicated element read through `SpmScalar::uniform_value`.
#[device(chip = 1)]
pub fn uniform_scalar_operand(device: &mut Device, input: &Vector) -> Vector {
    let values: SpmTensor<i32, Chip, Cluster, Pe0, m![E]> = input.to_spm(&mut device.tdma);
    let base = values.get(0u32).uniform_value();
    let mut result = SpmTensor::<i32, Chip, Cluster, Pe0, m![E]>::zero();
    device.per_cluster((&values, &mut result), |_, (values, mut result)| {
        for i in 0..result.len() {
            result.set(i, values.get(i) - base);
        }
    });
    result.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn update(device: &mut Device, input: &Vector, scale: &HbmScalar<i32>) -> Vector {
    let scale = scale.to_spm(&mut device.tdma);
    let mut values: SpmTensor<i32, Chip, Cluster, Pe0, m![E]> = input.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, values.get(i) + scale);
        }
    });
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn dma_distributed(device: &mut Device, input: &Distributed) -> Distributed {
    let values: SpmTensor<i32, Chip, Cluster, m![P], m![E]> = input.to_spm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn dma_replicated(device: &mut Device, input: &Vector) -> Vector {
    let values: SpmTensor<i32, Chip, Cluster, m![4], m![E]> = input.to_spm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn dma_dm(device: &mut Device, input: &Distributed) -> Distributed {
    let dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = input.to_dm(&mut device.tdma);
    let spm: SpmTensor<i32, Chip, m![1 # 2], m![P], m![E]> = dm.to_spm(&mut device.tdma);
    let dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = spm.to_dm(&mut device.tdma);
    dm.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn dma_redistribute(device: &mut Device, input: &Distributed) -> Distributed {
    let spm: SpmTensor<i32, Chip, Cluster, m![P], m![E]> = input.to_spm(&mut device.tdma);
    let spm: SpmTensor<i32, Chip, Cluster, Pe0, m![P, E]> = spm.to_spm(&mut device.tdma);
    let spm: SpmTensor<i32, Chip, Cluster, m![P], m![E]> = spm.to_spm(&mut device.tdma);
    spm.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn scalar_to_distributed(device: &mut Device, scale: &HbmScalar<i32>) -> Distributed {
    let scale = scale.to_spm(&mut device.tdma);
    let mut values = SpmTensor::<i32, Chip, Cluster, Pe0, m![P, E]>::zero();
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, scale + i as i32);
        }
    });
    let values: SpmTensor<i32, Chip, Cluster, m![P], m![E]> = values.to_spm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn distributed_to_scalar(device: &mut Device, input: &Distributed) -> Distributed {
    let values: SpmTensor<i32, Chip, Cluster, m![P], m![E]> = input.to_spm(&mut device.tdma);
    let mut values: SpmTensor<i32, Chip, Cluster, Pe0, m![P, E]> = values.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, values.get(i) + 7);
        }
    });
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn scalar_to_replicated(device: &mut Device, input: &Vector) -> Vector {
    let mut values: SpmTensor<i32, Chip, Cluster, Pe0, m![E]> = input.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, values.get(i) + 23);
        }
    });
    let values: SpmTensor<i32, Chip, Cluster, m![4], m![E]> = values.to_spm(&mut device.tdma);
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn replicated_to_scalar(device: &mut Device, input: &Vector) -> Vector {
    let values: SpmTensor<i32, Chip, Cluster, m![4], m![E]> = input.to_spm(&mut device.tdma);
    let mut values: SpmTensor<i32, Chip, Cluster, Pe0, m![E]> = values.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, values.get(i) + 31);
        }
    });
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn clustered_scalar_update(device: &mut Device, input: &Clustered, scale: &HbmScalar<i32>) -> Clustered {
    let scale = scale.to_spm(&mut device.tdma);
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P, E]> = input.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, values.get(i) * 3 + scale);
        }
    });
    values.to_hbm(&mut device.tdma)
}

pub type LocalLengths = HbmTensor<i32, Chip, m![C, P]>;

#[device(chip = 1)]
pub fn clustered_varying_index(device: &mut Device, lengths: &LocalLengths) -> LocalLengths {
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = lengths.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        let index = values.get(0u32) as u32;
        let old = values.get(index);
        values.set(index, old + 700);
    });
    values.to_hbm(&mut device.tdma)
}

/// Uses a 64-bit index for scalar SPM accesses.
#[device(chip = 1)]
pub fn subtract_into_new_spm(
    device: &mut Device,
    input: &LocalLengths,
    offset: &HbmScalar<i32>,
) -> (LocalLengths, LocalLengths) {
    let offset = offset.to_spm(&mut device.tdma);
    let source: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = input.to_spm(&mut device.tdma);
    let mut result = SpmTensor::<i32, Chip, m![C], Pe0, m![P]>::zero();
    device.per_cluster((&source, &mut result), |_, (source, mut result)| {
        for i in 0..result.len() {
            result.set(i, source.get(i) - offset);
        }
    });
    (source.to_hbm(&mut device.tdma), result.to_hbm(&mut device.tdma))
}

#[device(chip = 1)]
pub fn reshape_pe0_update(device: &mut Device, input: &Clustered) -> Clustered {
    let values: SpmTensor<i32, Chip, m![C], m![P], m![E]> = input.to_spm(&mut device.tdma);
    let mut values = unsafe { values.reshape::<Chip, m![C], Pe0, m![E]>() };
    device.per_cluster(&mut values, |_, mut values| {
        for i in 0..values.len() {
            values.set(i, values.get(i) + 1000);
        }
    });
    let values = unsafe { values.reshape::<Chip, m![C], m![P], m![E]>() };
    values.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn dma_core_reduce_pes(device: &mut Device, input: &Clustered) -> Clustered {
    let distributed: SpmTensor<i32, Chip, m![C], m![P], m![E]> = input.to_spm(&mut device.tdma);
    let source: SpmTensor<i32, Chip, m![C], Pe0, m![P, E]> = distributed.to_spm(&mut device.tdma);
    let mut local = SpmTensor::<i32, Chip, m![C], Pe0, m![E]>::zero();
    device.per_cluster((&source, &mut local), |_, (source, mut local)| {
        for i in 0..local.len() {
            local.set(
                i,
                source.get(i) + source.get(16 + i) + source.get(32 + i) + source.get(48 + i),
            );
        }
    });
    let replicated: SpmTensor<i32, Chip, m![C], m![4], m![E]> = local.to_spm(&mut device.tdma);
    let distributed = unsafe { replicated.reshape::<Chip, m![C], m![P], m![E]>() };
    distributed.to_hbm(&mut device.tdma)
}

/// Applies the bias stored in slot 0 of each cluster.
#[device(chip = 1)]
pub fn varying_scalar_operand(device: &mut Device, lengths: &LocalLengths) -> LocalLengths {
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = lengths.to_spm(&mut device.tdma);
    let bias = values.get(0u32);
    device.per_cluster((&bias, &mut values), |_, (bias, mut values)| {
        values.set(1u32, values.get(1u32) + bias);
        values.set(2u32, values.get(2u32) - bias);
    });
    values.to_hbm(&mut device.tdma)
}

/// Reads an HBM region into a fresh HBM tensor.
#[device(chip = 1)]
pub fn dma_hbm_view_to_hbm(device: &mut Device, input: &Distributed) -> Distributed {
    input.view().to_hbm(&mut device.tdma)
}

/// Reads a scratchpad region into a fresh DM tensor.
#[device(chip = 1)]
pub fn dma_spm_view_to_dm(device: &mut Device, input: &Distributed) -> Distributed {
    let spm: SpmTensor<i32, Chip, m![1 # 2], m![P], m![E]> = input.to_spm(&mut device.tdma);
    let dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = spm.view().to_dm(&mut device.tdma);
    dm.to_hbm(&mut device.tdma)
}

/// Writes a scratchpad region into an existing DM region.
#[device(chip = 1)]
pub fn dma_spm_view_to_dm_view(device: &mut Device, input: &Distributed) -> Distributed {
    let spm: SpmTensor<i32, Chip, m![1 # 2], m![P], m![E]> = input.to_spm(&mut device.tdma);
    let mut dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = DmTensor::new();
    spm.view().to_dm_view(&mut device.tdma, dm.view_mut());
    dm.to_hbm(&mut device.tdma)
}

/// Reads a DM region into a fresh HBM tensor.
#[device(chip = 1)]
pub fn dma_dm_view_to_hbm(device: &mut Device, input: &Distributed) -> Distributed {
    let dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = input.to_dm(&mut device.tdma);
    dm.view().to_hbm(&mut device.tdma)
}

/// Reads a DM region into a fresh DM tensor.
#[device(chip = 1)]
pub fn dma_dm_view_to_dm(device: &mut Device, input: &Distributed) -> Distributed {
    let dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = input.to_dm(&mut device.tdma);
    let dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = dm.view().to_dm(&mut device.tdma);
    dm.to_hbm(&mut device.tdma)
}

/// Writes a DM region into an existing scratchpad region.
#[device(chip = 1)]
pub fn dma_dm_view_to_spm_view(device: &mut Device, input: &Distributed) -> Distributed {
    let dm: DmTensor<i32, Chip, m![1 # 2], m![P, 64], m![E]> = input.to_dm(&mut device.tdma);
    let mut spm: SpmTensor<i32, Chip, m![1 # 2], m![P], m![E]> = input.to_spm(&mut device.tdma);
    dm.view().to_spm_view(&mut device.tdma, spm.view_mut());
    spm.to_hbm(&mut device.tdma)
}

/// Uses a host scalar to select the same write branch in every cluster.
#[device(chip = 1)]
pub fn uniform_scalar_branch(device: &mut Device, input: &LocalLengths, condition: &HbmScalar<bool>) -> LocalLengths {
    let condition = condition.to_spm(&mut device.tdma);
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = input.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        if condition {
            values.set(1u32, values.get(1u32) + 100);
        } else {
            values.set(2u32, values.get(2u32) + 200);
        }
    });
    values.to_hbm(&mut device.tdma)
}

/// Passes a host scalar to each cluster as an `SpmScalar` built by `SpmScalar::uniform`.
#[device(chip = 1)]
pub fn uniform_scalar_seed(device: &mut Device, input: &LocalLengths, scale: &HbmScalar<i32>) -> LocalLengths {
    let scale = scale.to_spm(&mut device.tdma);
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = input.to_spm(&mut device.tdma);
    let seed = SpmScalar::<i32, Chip, m![C]>::uniform(scale);
    let bias = SpmScalar::<i32, Chip, m![C]>::uniform(100);
    device.per_cluster((&seed, &bias, &mut values), |_, (seed, bias, mut values)| {
        values.set(1u32, values.get(1u32) + seed);
        values.set(2u32, values.get(2u32) + bias);
    });
    values.to_hbm(&mut device.tdma)
}

/// Sums two slots in each cluster, returns the sum as an `SpmScalar`, and writes it in a second body.
#[device(chip = 1)]
pub fn scalar_result(device: &mut Device, input: &LocalLengths) -> LocalLengths {
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = input.to_spm(&mut device.tdma);
    let total: SpmScalar<i32, Chip, m![C]> =
        device.per_cluster(&values, |_, values| values.get(1u32) + values.get(2u32));
    device.per_cluster((&total, &mut values), |_, (total, mut values)| {
        values.set(3u32, total);
    });
    values.to_hbm(&mut device.tdma)
}

/// Seeds a scalar from a host value and returns each cluster's updated total.
#[device(chip = 1)]
pub fn uniform_accumulator(device: &mut Device, input: &LocalLengths) -> LocalLengths {
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = input.to_spm(&mut device.tdma);
    let total = SpmScalar::<i32, Chip, m![C]>::uniform(100);
    let total: SpmScalar<i32, Chip, m![C]> = device.per_cluster((&values, &total), |_, (values, total)| {
        total + values.get(0u32) + values.get(1u32)
    });
    device.per_cluster((&total, &mut values), |_, (total, mut values)| {
        values.set(3u32, total);
    });
    values.to_hbm(&mut device.tdma)
}

/// Returns two scalars from one body: each cluster's own slot and a value shared by all clusters.
#[device(chip = 1)]
pub fn scalar_result_pair(device: &mut Device, input: &LocalLengths) -> LocalLengths {
    let mut values: SpmTensor<i32, Chip, m![C], Pe0, m![P]> = input.to_spm(&mut device.tdma);
    let (own, shared): (SpmScalar<i32, Chip, m![C]>, SpmScalar<i32, Chip, m![2]>) =
        device.per_cluster(&values, |_, values| (values.get(0u32) * 2, 7));
    let shared = shared.uniform_value();
    device.per_cluster((&own, &mut values), |_, (own, mut values)| {
        values.set(2u32, own + shared);
    });
    values.to_hbm(&mut device.tdma)
}

/// Clamps each length into `0..=capacity` with `max` and `min`, then replicates the window to every PE.
#[device(chip = 1)]
pub fn gather_window_lengths(
    device: &mut Device,
    input: &LocalLengths,
    begin: &HbmScalar<i32>,
    capacity: &HbmScalar<i32>,
) -> (LocalLengths, HbmTensor<i32, Chip, m![LengthClusters, LengthPes, C, P]>) {
    let begin = begin.to_spm(&mut device.tdma);
    let capacity = capacity.to_spm(&mut device.tdma);
    let source: SpmTensor<i32, Chip, m![2], Pe0, m![C, P]> = input.to_spm(&mut device.tdma);
    let mut window = SpmTensor::<i32, Chip, m![2], Pe0, m![C, P]>::zero();
    device.per_cluster((&source, &mut window), |_, (source, mut window)| {
        for i in 0..window.len() {
            window.set(i, (source.get(i) - begin).max(0).min(capacity));
        }
    });
    let replicated: SpmTensor<i32, Chip, m![2], m![4], m![C, P]> = window.to_spm(&mut device.tdma);
    let replicas = unsafe { replicated.reshape::<Chip, m![LengthClusters], m![LengthPes], m![C, P]>() };
    (source.to_hbm(&mut device.tdma), replicas.to_hbm(&mut device.tdma))
}
