//! A device function loads on its first launch; that load must not enter an executor, so the first
//! launch works under any executor, a blocking one included.
#![cfg(backend = "npu")]

use furiosa_opt_examples::pe_count::{P4, four_pe_add};
use furiosa_opt_std::prelude::*;

/// The first launch of a function, which loads it, completes inside `futures_executor::block_on`.
#[test]
fn loads_under_a_blocking_executor() -> eyre::Result<()> {
    futures_executor::block_on(async {
        let mut device = Device::new(four_pe_add.topology())?;
        let input = HostTensor::<i32, m![P4]>::from_vec((0..<m![P4]>::SIZE as i32).collect::<Vec<_>>());
        let input_hbm = input.to_hbm(&mut device.pdma).await?;
        let out = launch(four_pe_add, (&mut device, &input_hbm)).await?;
        let expected = (0..<m![P4]>::SIZE as i32).map(|x| x + 1).collect::<Vec<_>>();
        assert_eq!(out.to_host::<m![P4]>(&mut device.pdma).await?.into_vec(), expected);
        Ok(())
    })
}
