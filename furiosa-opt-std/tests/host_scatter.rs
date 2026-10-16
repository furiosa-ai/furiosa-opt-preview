use furiosa_mapping::*;
use furiosa_opt_std::backend::{Cpu, IndexUnit};
use furiosa_opt_std::prelude::Tensor;
use std::num::NonZeroUsize;

axes![Pool = 512, Rows = 128, Kv = 2, Heads = 8, Width = 128];

#[test]
fn scatters_replicated_rows() {
    let source = Tensor::<i32, m![Rows, Kv, Heads, Width], Cpu>::from_vec((0..128 * 2048).map(|at| at % 251));
    type Updates = Tensor<i32, m![2, Heads, 1 # 32, Rows, Kv, Width], Cpu>;
    let updates: Updates = source.transpose(true);
    let offsets = Tensor::<i32, m![Rows], Cpu>::from_vec((256..384).map(|row| row * 8192));
    let mut destination = Tensor::<i32, m![Pool, Kv, Heads, Width], Cpu>::splat(-1);

    updates.scatter::<m![Rows], m![Pool], _, _>(
        &mut destination,
        &offsets,
        &<m![1]>::to_value(),
        IndexUnit::Bytes(NonZeroUsize::new(8192).unwrap()),
    );

    let expected = (0..512 * 2048)
        .map(|at| {
            if (256..384).contains(&(at / 2048)) {
                (at - 256 * 2048) % 251
            } else {
                -1
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(destination.into_vec(), expected);
}
