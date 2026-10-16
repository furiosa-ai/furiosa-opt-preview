//! A weight-streaming loop bounded by a device-tensor count produces, for the groups it names, what
//! the unrolled loop over every group produces, whether its body holds one group or a pair.

mod extent_stream {
    use furiosa_opt_examples::extent_stream::{
        Group, One, Out, Red, Tok, counted_groups, counted_pairs, unrolled_groups,
    };
    use furiosa_opt_std::prelude::*;

    /// Launches the three kernels on the same operands, the counted one over `groups` groups and
    /// the paired one over `pairs` pairs, and returns every group of each: (unrolled, counted,
    /// paired).
    async fn stream_groups(groups: i32, pairs: i32) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
        let mut ctx = Device::new(unrolled_groups.topology()).unwrap();
        let group_count = HostTensor::<i32, m![One]>::from_vec([groups])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await
            .unwrap();
        let pair_count = HostTensor::<i32, m![One]>::from_vec([pairs])
            .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
            .await
            .unwrap();
        let activation = HostTensor::<bf16, m![Tok, Red]>::from_vec(
            (0..<m![Tok, Red]>::SIZE)
                .map(|i| bf16::from_f32(((i / Red::SIZE * 7 + i % Red::SIZE) % 3 == 0) as u8 as f32))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![Tok, Red]>(&mut ctx.pdma)
        .await
        .unwrap();
        let weight = HostTensor::<bf16, m![Group, Out, Red]>::from_vec(
            (0..<m![Group, Out, Red]>::SIZE)
                .map(|i| bf16::from_f32(((i / Red::SIZE * 5 + i % Red::SIZE) % 2) as f32))
                .collect::<Vec<_>>(),
        )
        .to_hbm::<m![1], m![Group, Out, Red]>(&mut ctx.pdma)
        .await
        .unwrap();

        let unrolled = launch(unrolled_groups, (&mut ctx, &activation, &weight))
            .await
            .unwrap()
            .to_host::<m![Tok, Group, Out]>(&mut ctx.pdma)
            .await
            .unwrap()
            .into_vec();
        let counted = launch(counted_groups, (&mut ctx, &group_count, &activation, &weight))
            .await
            .unwrap()
            .to_host::<m![Tok, Group, Out]>(&mut ctx.pdma)
            .await
            .unwrap()
            .into_vec();
        let paired = launch(counted_pairs, (&mut ctx, &pair_count, &activation, &weight))
            .await
            .unwrap()
            .to_host::<m![Tok, Group, Out]>(&mut ctx.pdma)
            .await
            .unwrap()
            .into_vec();
        (unrolled, counted, paired)
    }

    /// The leading `groups` groups of every token.
    fn leading(stream: &[bf16], groups: usize) -> Vec<bf16> {
        stream
            .chunks(Group::SIZE * Out::SIZE)
            .flat_map(|token| token[..groups * Out::SIZE].to_vec())
            .collect()
    }

    #[tokio::test]
    async fn counted_matches_unrolled_when_the_count_fills_the_envelope() {
        let (unrolled, counted, paired) = stream_groups(Group::SIZE as i32, (Group::SIZE / 2) as i32).await;
        let zero = bf16::from_f32(0.0);
        assert!(
            unrolled.iter().any(|&value| value != zero),
            "the fixture produced nothing"
        );
        assert_eq!(counted, unrolled);
        assert_eq!(paired, unrolled);
    }

    #[tokio::test]
    async fn counted_matches_unrolled_on_the_leading_groups_inside_the_envelope() {
        let (unrolled, counted, paired) = stream_groups(3, 2).await;
        assert_eq!(leading(&counted, 3).len(), Tok::SIZE * 3 * Out::SIZE);
        assert_eq!(leading(&counted, 3), leading(&unrolled, 3));
        assert_eq!(leading(&paired, 4), leading(&unrolled, 4));
    }
}
