#![feature(register_tool, proc_macro_hygiene)]
#![register_tool(furiosa_opt)]

use furiosa_opt_examples::scatter_gather::{D, G, K};
use furiosa_opt_std::prelude::*;

#[device(chip = 1)]
fn gather(
    device: &mut Device,
    table: &HbmTensor<bf16, m![1], m![K, D]>,
    index: &HbmTensor<i32, m![1], m![G]>,
    lengths: &HbmTensor<i32, m![1], m![1]>,
) -> HbmTensor<bf16, m![1], m![G, D]> {
    table
        .gather::<m![K], m![D]>()
        .by_byte_offsets(index)
        .prefix::<m![G], _>(lengths, 0)
        .to_dm::<m![1 # 2], m![G / 2], m![G % 2, D]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

#[tokio::test]
async fn sparse_prefix_skips_unexecuted_indices() -> eyre::Result<()> {
    let mut device = Device::new(gather.topology())?;
    let table = HostTensor::<bf16, m![K, D]>::from_vec(vec![bf16::from_f32(3.0); K::SIZE * D::SIZE])
        .to_hbm(&mut device.pdma)
        .await?;
    for length in [G::SIZE as i32, 1, 0, -1] {
        let requested = length.clamp(0, G::SIZE as i32) as usize;
        let bound = requested.saturating_add(63).min(G::SIZE);
        let index = HostTensor::<i32, m![G]>::from_vec((0..G::SIZE).map(|at| if at < bound { 0 } else { i32::MAX }))
            .to_hbm(&mut device.pdma)
            .await?;
        let valid_length = HostTensor::<i32, m![1]>::from_vec([length])
            .to_hbm(&mut device.pdma)
            .await?;
        let output = launch(gather, (&mut device, &table, &index, &valid_length)).await?;
        let values = output.to_host::<m![G, D]>(&mut device.pdma).await?.into_vec();
        assert!(
            values[..requested * D::SIZE].iter().all(|v| *v == bf16::from_f32(3.0)),
            "length {length}"
        );
    }
    Ok(())
}
