//! A loop bounded by a device-tensor count runs exactly that many iterations of the `Live` it is
//! allocated for, so the tiles beyond the count stay zero.

mod dynamic_extent {
    use furiosa_opt_examples::dynamic_extent::{Live, One, Wide, copy_live_tiles};
    use furiosa_opt_std::prelude::*;

    /// Launches the copy with `live_tiles` on the count tensor and returns the elements it copied.
    async fn copied_elements(live_tiles: i32) -> usize {
        let mut ctx = Device::new(copy_live_tiles.topology()).unwrap();
        let live = HostTensor::<i32, m![One]>::from_vec([live_tiles])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await
            .unwrap();
        let input = HostTensor::<bf16, m![Wide]>::from_vec(vec![bf16::from_f32(1.0); Wide::SIZE])
            .to_hbm::<m![1], m![Wide]>(&mut ctx.pdma)
            .await
            .unwrap();

        launch(copy_live_tiles, (&mut ctx, &live, &input))
            .await
            .unwrap()
            .to_host::<m![Wide]>(&mut ctx.pdma)
            .await
            .unwrap()
            .into_vec()
            .iter()
            .filter(|value| value.to_f32() == 1.0)
            .count()
    }

    const TILE: usize = Wide::SIZE / Live::SIZE;

    #[tokio::test]
    async fn copies_nothing_for_a_zero_count() {
        assert_eq!(copied_elements(0).await, 0);
    }

    #[tokio::test]
    async fn copies_one_tile_for_a_count_of_one() {
        assert_eq!(copied_elements(1).await, TILE);
    }

    #[tokio::test]
    async fn copies_the_counted_tiles_inside_the_envelope() {
        assert_eq!(copied_elements(3).await, 3 * TILE);
    }

    #[tokio::test]
    async fn copies_every_tile_when_the_count_fills_the_envelope() {
        assert_eq!(copied_elements(Live::SIZE as i32).await, Wide::SIZE);
    }
}

/// The count belongs to the row the outer loop is on, not to the tensor's first element.
///
/// Only the second row has work, so a count read once for the whole tensor would copy nothing,
/// and a count read per row copies that row's three tiles.
#[tokio::test]
async fn copies_each_row() -> eyre::Result<()> {
    use furiosa_opt_examples::dynamic_extent::{Access, One, Rows, Wide, copy_live_tiles_per_row};
    use furiosa_opt_std::prelude::*;
    let mut device = Device::new(copy_live_tiles_per_row.topology())?;
    let mut counts = vec![0i32; Rows::SIZE * Access::SIZE];
    counts[Access::SIZE] = 3;

    let live = HostTensor::<i32, m![Rows, Access]>::from_vec(counts)
        .to_hbm::<m![1], m![Rows, Access]>(&mut device.pdma)
        .await?;
    let rows = HostTensor::<i32, m![One]>::from_vec([Rows::SIZE as i32])
        .to_hbm::<m![1], m![One]>(&mut device.pdma)
        .await?;
    let input = HostTensor::<bf16, m![Wide]>::from_vec(vec![bf16::from_f32(1.0); Wide::SIZE])
        .to_hbm::<m![1], m![Wide]>(&mut device.pdma)
        .await?;
    let mut output = HostTensor::<bf16, m![Wide]>::zero()
        .to_hbm::<m![1], m![Wide]>(&mut device.pdma)
        .await?;

    launch(
        copy_live_tiles_per_row,
        (&mut device, &rows, &live, &input, &mut output),
    )
    .await?;

    let copied = output
        .to_host::<m![Wide]>(&mut device.pdma)
        .await?
        .into_vec()
        .iter()
        .filter(|value| value.to_f32() == 1.0)
        .count();
    assert_eq!(copied, 3 * 512);
    Ok(())
}

#[tokio::test]
async fn preserves_excluded_tiles() -> eyre::Result<()> {
    use furiosa_opt_examples::dynamic_extent::{Live, One, Wide, copy_remaining};
    use furiosa_opt_std::prelude::*;

    for count in [-1, 0, 1, 2, 7, 16, 20] {
        let mut device = Device::new(copy_remaining.topology())?;
        let count_tensor = HostTensor::<i32, m![One]>::from_vec([count])
            .to_hbm::<m![1], m![One]>(&mut device.pdma)
            .await?;
        let input = HostTensor::<bf16, m![Wide]>::from_vec(vec![bf16::from_f32(1.0); Wide::SIZE])
            .to_hbm::<m![1], m![Wide]>(&mut device.pdma)
            .await?;
        let mut output = HostTensor::<bf16, m![Wide]>::from_vec(vec![bf16::from_f32(-1.0); Wide::SIZE])
            .to_hbm::<m![1], m![Wide]>(&mut device.pdma)
            .await?;

        launch(copy_remaining, (&mut device, &count_tensor, &input, &mut output)).await?;

        let expected: Vec<_> = (0..Wide::SIZE)
            .map(|at| {
                let tile = at / 512;
                bf16::from_f32(
                    if tile >= 1 && tile < usize::try_from(count).unwrap_or(0).min(Live::SIZE) {
                        1.0
                    } else {
                        -1.0
                    },
                )
            })
            .collect();
        assert_eq!(
            output.to_host::<m![Wide]>(&mut device.pdma).await?.into_vec(),
            expected,
            "count {count}"
        );
    }
    Ok(())
}
