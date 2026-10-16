//! Two nested loops, each bounded by its own device-tensor count, touch exactly the tiles the two
//! counts name out of the `Outer` x `Inner` they are allocated for.

mod nested_extent {
    use furiosa_opt_examples::nested_extent::{Inner, One, Outer, Wide, copy_live_grid};
    use furiosa_opt_std::prelude::*;

    const TILE: usize = Wide::SIZE / (Outer::SIZE * Inner::SIZE);

    /// Launches the copy with the two counts and returns the tiles it touched, in order. Tile `t`
    /// of the input holds `t + 1` and the output starts out all `-1`, so a copied tile is told
    /// apart from one the kernel never wrote.
    async fn copied_tiles(outers: i32, inners: i32) -> Vec<usize> {
        let mut ctx = Device::new(copy_live_grid.topology()).unwrap();
        let outers = HostTensor::<i32, m![One]>::from_vec([outers])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await
            .unwrap();
        let inners = HostTensor::<i32, m![One]>::from_vec([inners])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await
            .unwrap();
        let input = HostTensor::<bf16, m![Wide]>::from_vec(
            (0..Wide::SIZE)
                .map(|i| bf16::from_f32((i / TILE + 1) as f32))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![Wide]>(&mut ctx.pdma)
        .await
        .unwrap();
        let untouched = bf16::from_f32(-1.0);
        let mut output = HostTensor::<bf16, m![Wide]>::from_vec(vec![untouched; Wide::SIZE])
            .to_hbm::<m![1], m![Wide]>(&mut ctx.pdma)
            .await
            .unwrap();

        launch(copy_live_grid, (&mut ctx, &outers, &inners, &input, &mut output))
            .await
            .unwrap();
        output
            .to_host::<m![Wide]>(&mut ctx.pdma)
            .await
            .unwrap()
            .into_vec()
            .chunks(TILE)
            .enumerate()
            .filter_map(|(tile, values)| {
                let copied = bf16::from_f32((tile + 1) as f32);
                if values.iter().all(|&value| value == copied) {
                    return Some(tile);
                }
                assert!(
                    values.iter().all(|&value| value == untouched),
                    "tile {tile} is neither copied nor untouched"
                );
                None
            })
            .collect()
    }

    /// The tiles the counts name: every `(outer, inner)` pair below both counts.
    fn grid(outers: usize, inners: usize) -> Vec<usize> {
        (0..outers)
            .flat_map(|outer| (0..inners).map(move |inner| outer * Inner::SIZE + inner))
            .collect()
    }

    #[tokio::test]
    async fn copies_nothing_for_a_zero_outer_count() {
        assert_eq!(copied_tiles(0, Inner::SIZE as i32).await, grid(0, 0));
    }

    #[tokio::test]
    async fn copies_nothing_for_a_zero_inner_count() {
        assert_eq!(copied_tiles(Outer::SIZE as i32, 0).await, grid(0, 0));
    }

    #[tokio::test]
    async fn copies_one_tile_for_counts_of_one() {
        assert_eq!(copied_tiles(1, 1).await, grid(1, 1));
    }

    #[tokio::test]
    async fn copies_the_leading_inner_tiles_of_the_leading_outers() {
        assert_eq!(copied_tiles(2, 3).await, grid(2, 3));
    }

    #[tokio::test]
    async fn copies_every_tile_when_both_counts_fill_their_envelopes() {
        assert_eq!(
            copied_tiles(Outer::SIZE as i32, Inner::SIZE as i32).await,
            grid(Outer::SIZE, Inner::SIZE)
        );
    }
}
