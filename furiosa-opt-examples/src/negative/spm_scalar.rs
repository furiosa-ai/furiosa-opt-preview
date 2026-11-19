//! Rejected scratchpad scalar accesses.

use furiosa_opt_std::prelude::*;
axes![C = 2, E = 4];
type Input = HbmTensor<i32, m![1], m![C, E]>;
/// Replicated PE storage cannot be updated through a PE 0 scalar write.
#[device(chip = 1)]
pub fn nonlocal_pe(device: &mut Device, input: &Input) -> Input {
    let mut values: SpmTensor<i32, m![1], m![C], m![4], m![E]> = input.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        values.set(0usize, 1);
    });
    values.to_hbm(&mut device.tdma)
}

/// An unguarded panic that MIR-to-vISA must reject before graph pruning removes it.
#[device(chip = 1)]
pub fn unguarded_panic(device: &mut Device, input: &Input) -> Input {
    let mut values: SpmTensor<i32, m![1], m![C], m![1 # 4], m![E]> = input.to_spm(&mut device.tdma);
    device.per_cluster(&mut values, |_, mut values| {
        values.set(1u32, 7);
        panic!("nothing guards this");
    });
    values.to_hbm(&mut device.tdma)
}

/// A value that differs by cluster cannot be written into a tensor whose `Cluster` broadcasts.
#[device(chip = 1)]
pub fn varying_write_to_broadcast(device: &mut Device, input: &Input) -> HbmTensor<i32, m![1], m![E]> {
    let source: SpmTensor<i32, m![1], m![C], m![1 # 4], m![E]> = input.to_spm(&mut device.tdma);
    let mut shared = SpmTensor::<i32, m![1], m![2], m![1 # 4], m![E]>::zero();
    device.per_cluster((&source, &mut shared), |_, (source, mut shared)| {
        shared.set(0u32, source.get(0u32));
    });
    shared.to_hbm(&mut device.tdma)
}
