//! A loop bounded by a device-tensor count carries its running sum across exactly that many
//! iterations of the `Live` it is allocated for.

mod extent_accumulate {
    use furiosa_opt_examples::extent_accumulate::{Live, One, W, sum_live_rows};
    use furiosa_opt_std::prelude::*;

    /// Launches the sum with `live` on the count tensor and returns the result. Row `r` of the input
    /// holds `r + 1` throughout and the initial sum holds its element index, so the result says
    /// which rows were added and that the initial value survived.
    async fn summed(live: i32) -> Vec<i32> {
        let mut ctx = Device::new(sum_live_rows.topology()).unwrap();
        let live = HostTensor::<i32, m![One]>::from_vec([live])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await
            .unwrap();
        let init = HostTensor::<i32, m![W]>::from_vec((0..W::SIZE as i32).collect::<Vec<_>>())
            .to_hbm::<m![1], m![W]>(&mut ctx.pdma)
            .await
            .unwrap();
        let input = HostTensor::<i32, m![Live, W]>::from_vec(
            (0..Live::SIZE * W::SIZE)
                .map(|i| (i / W::SIZE + 1) as i32)
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![Live, W]>(&mut ctx.pdma)
        .await
        .unwrap();

        launch(sum_live_rows, (&mut ctx, &live, &init, &input))
            .await
            .unwrap()
            .to_host::<m![W]>(&mut ctx.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// Element `j` after adding rows `1..=live` onto an initial `j`.
    fn expected(live: i32) -> Vec<i32> {
        (0..W::SIZE as i32).map(|j| j + live * (live + 1) / 2).collect()
    }

    #[tokio::test]
    async fn keeps_the_initial_sum_for_a_zero_count() {
        assert_eq!(summed(0).await, expected(0));
    }

    #[tokio::test]
    async fn adds_one_row_for_a_count_of_one() {
        assert_eq!(summed(1).await, expected(1));
    }

    #[tokio::test]
    async fn adds_the_counted_rows_inside_the_envelope() {
        assert_eq!(summed(3).await, expected(3));
    }

    #[tokio::test]
    async fn adds_every_row_when_the_count_fills_the_envelope() {
        assert_eq!(summed(Live::SIZE as i32).await, expected(Live::SIZE as i32));
    }
}
