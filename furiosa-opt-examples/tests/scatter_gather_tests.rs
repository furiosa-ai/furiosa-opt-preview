//! Tests for scatter/gather DMA operations.
//!
//! Equivalent PyTorch (K = Scatter_key, D = Payload, C = Cache_len):
//! ```python
//! cache = torch.zeros(C, D, dtype=torch.bfloat16)  # [612, 128]
//! data = torch.randn(K, D, dtype=torch.bfloat16)   # [512, 128]
//! index = torch.arange(K)                          # [512]
//! cache[index] = data                              # index_put along dim 0
//! assert (cache[:K] != 0).all()
//! assert (cache[K:] == 0).all()
//! ```

use furiosa_opt_examples::scatter_gather::{
    Block, Blocks, Head, Heads, Table, Updates, gather_pool_blocks, scatter_minimal, scatter_pool_rows,
};
use furiosa_opt_std::prelude::*;

type Pool = m![Blocks, Block, Heads, Head];
type Updated = m![Updates, Heads, Head];

/// Blocks the block table names, out of order and from both ends of the pool, so a table entry
/// read as a row (or its block number scaled by the wrong byte pitch) lands on other rows.
const BLOCK_TABLE: [usize; <m![Table]>::SIZE] = [5, 0, 7, 2];

/// Slot of update row `u`: a non-identity permutation (`37` is coprime to the row count) that
/// spreads the rows over every block, so a scatter axis spanning one block cannot hold them.
fn slot(u: usize) -> usize {
    (u * 37 + 5) % <m![Blocks, Block]>::SIZE
}

#[tokio::test]
async fn test_scatter_minimal() -> eyre::Result<()> {
    use furiosa_opt_examples::scatter_gather::{C, D, K};

    let _ = env_logger::try_init();
    let mut device = Device::new(scatter_minimal.topology())?;

    let data: HbmTensor<bf16, m![1], m![K, D]> = HostTensor::<bf16, m![K, D]>::from_vec(
        (1..=<m![K, D]>::SIZE)
            .map(|i| bf16::from_f32(i as f32))
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut device.pdma)
    .await?;

    // Convert row indices to byte offsets (scaled=true).
    let entry_bytes = (<m![D]>::SIZE * std::mem::size_of::<bf16>()) as i32;
    let index: HbmTensor<i32, m![1], m![K]> =
        HostTensor::<i32, m![K]>::from_vec((0..<m![K]>::SIZE as i32).map(|i| i * entry_bytes).collect::<Vec<_>>())
            .to_hbm(&mut device.pdma)
            .await?;

    let mut output: HbmTensor<bf16, m![1], m![C, D]> =
        HostTensor::<bf16, m![C, D]>::zero().to_hbm(&mut device.pdma).await?;

    launch(scatter_minimal, (&mut device, &data, &index, &mut output)).await?;

    let actual = output.to_host::<m![C, D]>(&mut device.pdma).await?.into_vec();
    let expected = Tensor::<bf16, m![C, D]>::from_vec(
        (0..<m![C, D]>::SIZE)
            .map(|i| bf16::from_f32(if i < <m![K, D]>::SIZE { (i + 1) as f32 } else { 0.0 }))
            .collect::<Vec<_>>(),
    )
    .into_vec();

    assert_eq!(actual, expected);

    Ok(())
}

/// Rows scattered into a pool kept in blocks of rows: each lands in its slot whichever block holds
/// it, and every other row stays zero.
#[tokio::test]
async fn test_scatter_pool_rows() -> eyre::Result<()> {
    let _ = env_logger::try_init();
    let mut device = Device::new(scatter_pool_rows.topology())?;

    let row_len = <m![Heads, Head]>::SIZE;
    let update_vals: Vec<bf16> = (1..=<Updated>::SIZE).map(|i| bf16::from_f32(i as f32)).collect();
    let row_bytes = (row_len * std::mem::size_of::<bf16>()) as i32;
    let index_vals: Vec<i32> = (0..<m![Updates]>::SIZE).map(|u| slot(u) as i32 * row_bytes).collect();

    let updates: HbmTensor<bf16, m![1], Updated> = HostTensor::<bf16, Updated>::from_vec(update_vals.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let index: HbmTensor<i32, m![1], m![Updates]> = HostTensor::<i32, m![Updates]>::from_vec(index_vals)
        .to_hbm(&mut device.pdma)
        .await?;
    let mut pool: HbmTensor<bf16, m![1], Pool> = HostTensor::<bf16, Pool>::zero().to_hbm(&mut device.pdma).await?;

    launch(scatter_pool_rows, (&mut device, &updates, &index, &mut pool)).await?;

    let actual = pool.to_host::<Pool>(&mut device.pdma).await?.into_vec();
    let mut expected = vec![bf16::from_f32(0.0); <Pool>::SIZE];
    for u in 0..<m![Updates]>::SIZE {
        expected[slot(u) * row_len..][..row_len].copy_from_slice(&update_vals[u * row_len..][..row_len]);
    }
    assert_eq!(actual, expected);
    Ok(())
}

/// Blocks gathered by a block table: output `[t, h, r, :]` is row `r` of the block entry `t` names,
/// head `h`.
#[tokio::test]
async fn test_gather_pool_blocks() -> eyre::Result<()> {
    let _ = env_logger::try_init();
    let mut device = Device::new(gather_pool_blocks.topology())?;

    let pool_vals: Vec<bf16> = (1..=<Pool>::SIZE).map(|i| bf16::from_f32(i as f32)).collect();
    let block_bytes = (<m![Block, Heads, Head]>::SIZE * std::mem::size_of::<bf16>()) as i32;
    let table_vals: Vec<i32> = BLOCK_TABLE.iter().map(|&b| b as i32 * block_bytes).collect();

    let pool: HbmTensor<bf16, m![1], Pool> = HostTensor::<bf16, Pool>::from_vec(pool_vals.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let block_table: HbmTensor<i32, m![1], m![Table]> = HostTensor::<i32, m![Table]>::from_vec(table_vals)
        .to_hbm(&mut device.pdma)
        .await?;

    let gathered = launch(gather_pool_blocks, (&mut device, &pool, &block_table)).await?;

    let actual = gathered
        .to_host::<m![Table, Heads, Block, Head]>(&mut device.pdma)
        .await?
        .into_vec();
    let head_len = <m![Head]>::SIZE;
    let mut expected = Vec::with_capacity(<m![Table, Heads, Block, Head]>::SIZE);
    for &b in &BLOCK_TABLE {
        for h in 0..<m![Heads]>::SIZE {
            for r in 0..<m![Block]>::SIZE {
                let row = b * <m![Block]>::SIZE + r;
                expected.extend_from_slice(&pool_vals[(row * <m![Heads]>::SIZE + h) * head_len..][..head_len]);
            }
        }
    }
    assert_eq!(actual, expected);
    Ok(())
}
