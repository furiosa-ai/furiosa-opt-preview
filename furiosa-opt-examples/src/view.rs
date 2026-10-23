pub mod simpl {
    use furiosa_opt_std::prelude::*;

    axes![A = 512, B = 8];

    #[device(chip = 1)]
    pub fn view_simpl(
        device: &mut Device,
        input_hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        // Transfer input tensor to DM with shape:
        // - (A=512/256) on the cluster dimension,
        // - (A=512%256) on the slice dimension,
        // - (B=8) on the element dimension.
        let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B]>(&mut device.tdma);

        // Allocate output tensor in DM with the same shape as input.
        let mut output = DmTensor::<i32, m![1], m![A / 256], m![A % 256], m![B]>::new();

        // Create views on input (B=0,1,2,3) and output (B=2,3,4,5), and copy data from input to output.
        let input0123 = input.view().tile::<m![B], 4, m![B = 4 # 8]>(0);
        let output2345 = output.view_mut().tile::<m![B], 4, m![B = 4 #{!} 8]>(2);
        input0123.to_dm_view(&mut device.tdma, output2345);

        // Create views on input (B=4,5) and output (B=6,7), and copy data from input to output.
        let input45 = input.view().tile::<m![B], 2, m![B = 2 # 8]>(4);
        let output67 = output.view_mut().tile::<m![B], 2, m![B = 2 #{!} 8]>(6);
        input45.to_dm_view(&mut device.tdma, output67);

        // Create views on input (B=6,7) and output (B=0,1), and copy data from input to output.
        let input67 = input.view().tile::<m![B], 2, m![B = 2 # 8]>(6);
        let output01 = output.view_mut().tile::<m![B], 2, m![B = 2 #{!} 8]>(0);
        input67.to_dm_view(&mut device.tdma, output01);

        // Transfer output tensor back to HBM.
        output.to_hbm(&mut device.tdma)
    }
}

pub mod nested {
    use furiosa_opt_std::prelude::*;

    axes![A = 512, B = 528];

    #[device(chip = 1)]
    pub fn view_nested(
        device: &mut Device,
        input_hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B]>(&mut device.tdma);
        let mut output = DmTensor::<i32, m![1], m![A / 256], m![A % 256], m![B]>::new();

        // Define the whole output first (full copy) so no position is left `Uninit`.
        input.view().to_dm_view(&mut device.tdma, output.view_mut());

        let in_nested = input
            .view()
            .tile::<m![B], 264, m![B = 264 # 528]>(0)
            .tile::<m![B = 264 # 528], 132, m![B = 132 # 528]>(0);
        let out_nested = output
            .view_mut()
            .tile::<m![B], 264, m![B = 264 #{!} 528]>(0)
            .tile::<m![B = 264 #{!} 528], 132, m![B = 132 #{!} 528]>(0);
        in_nested.to_dm_view(&mut device.tdma, out_nested);

        output.to_hbm(&mut device.tdma)
    }
}

/// Tile widths that do not divide their axis: each window keeps its `Resize` through the DMA.
pub mod nondividing {
    use furiosa_opt_std::prelude::*;

    axes![A = 512, B = 8, C = 2, P = 64, K = 22, L = 8, R = 96];

    /// [`super::simpl::view_simpl`] with widths 3 + 5 of `B`, rotating it by 5. `C` keeps each SRAM
    /// run a multiple of 8 bytes.
    #[device(chip = 1)]
    pub fn view_nondividing(
        device: &mut Device,
        input_hbm: &HbmTensor<i32, m![1], m![A, B, C]>,
    ) -> HbmTensor<i32, m![1], m![A, B, C]> {
        let input = input_hbm.to_dm::<m![A / 256], m![A % 256], m![B, C]>(&mut device.tdma);
        let mut output = DmTensor::<i32, m![1], m![A / 256], m![A % 256], m![B, C]>::new();

        let input012 = input.view().tile::<m![B], 3, m![B = 3 # 8, C]>(0);
        let output567 = output.view_mut().tile::<m![B], 3, m![B = 3 #{!} 8, C]>(5);
        input012.to_dm_view(&mut device.tdma, output567);

        let input34567 = input.view().tile::<m![B], 5, m![B = 5 # 8, C]>(3);
        let output01234 = output.view_mut().tile::<m![B], 5, m![B = 5 #{!} 8, C]>(0);
        input34567.to_dm_view(&mut device.tdma, output01234);

        output.to_hbm(&mut device.tdma)
    }

    /// Splits the middle `K` unevenly (8 + 8 + 6) through DM and writes the chunks back reversed:
    /// `[16, 22)` to `[0, 6)`, `[8, 16)` to `[6, 14)`, `[0, 8)` to `[14, 22)`.
    #[device(chip = 1, pe = 1)]
    pub fn hbm_nondividing(
        device: &mut Device,
        input: &HbmTensor<i32, m![1], m![P, K, L]>,
    ) -> HbmTensor<i32, m![1], m![P, K, L]> {
        let mut output = HbmTensor::<i32, m![1], m![P, K, L]>::new();

        let k0: DmTensor<i32, m![1], m![1], m![P], m![K = 8, L]> = input
            .view()
            .tile::<m![K], 8, m![P, K = 8 # 22, L]>(0)
            .to_dm(&mut device.tdma);
        let k1: DmTensor<i32, m![1], m![1], m![P], m![K = 8, L]> = input
            .view()
            .tile::<m![K], 8, m![P, K = 8 # 22, L]>(8)
            .to_dm(&mut device.tdma);
        let k2: DmTensor<i32, m![1], m![1], m![P], m![K = 6, L]> = input
            .view()
            .tile::<m![K], 6, m![P, K = 6 # 22, L]>(16)
            .to_dm(&mut device.tdma);

        k2.view().to_hbm_view(
            &mut device.tdma,
            output.view_mut().tile::<m![K], 6, m![P, K = 6 #{!} 22, L]>(0),
        );
        k1.view().to_hbm_view(
            &mut device.tdma,
            output.view_mut().tile::<m![K], 8, m![P, K = 8 #{!} 22, L]>(6),
        );
        k0.view().to_hbm_view(
            &mut device.tdma,
            output.view_mut().tile::<m![K], 8, m![P, K = 8 #{!} 22, L]>(14),
        );

        output
    }

    /// Tiles the inner digit `R % 12` by 5 + 7, rotating every 12-row by 7.
    #[device(chip = 1)]
    pub fn inner_digit_nondividing(
        device: &mut Device,
        input: &HbmTensor<i32, m![1], m![R]>,
    ) -> HbmTensor<i32, m![1], m![R]> {
        let mut output = HbmTensor::<i32, m![1], m![R]>::new();

        let head = input.view().tile::<m![R % 12], 5, m![R / 12, R % 12 = 5 # 12]>(0);
        head.to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![R % 12], 5, m![R / 12, R % 12 = 5 #{!} 12]>(7),
        );
        let tail = input.view().tile::<m![R % 12], 7, m![R / 12, R % 12 = 7 # 12]>(5);
        tail.to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![R % 12], 7, m![R / 12, R % 12 = 7 #{!} 12]>(0),
        );

        output
    }

    /// Tiles the outer digit `R / 12` by 3 + 5 rows, rotating the rows by 5.
    #[device(chip = 1)]
    pub fn outer_digit_nondividing(
        device: &mut Device,
        input: &HbmTensor<i32, m![1], m![R]>,
    ) -> HbmTensor<i32, m![1], m![R]> {
        let mut output = HbmTensor::<i32, m![1], m![R]>::new();

        let head = input.view().tile::<m![R / 12], 3, m![R / 12 = 3 # 8, R % 12]>(0);
        head.to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![R / 12], 3, m![R / 12 = 3 #{!} 8, R % 12]>(5),
        );
        let tail = input.view().tile::<m![R / 12], 5, m![R / 12 = 5 # 8, R % 12]>(3);
        tail.to_hbm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![R / 12], 5, m![R / 12 = 5 #{!} 8, R % 12]>(0),
        );

        output
    }
}

pub mod padding {
    use furiosa_opt_std::prelude::*;

    axes![A = 9, B = 7];
}

pub mod whole_view_mut {
    use furiosa_opt_std::prelude::*;

    axes![S = 128];

    #[device(chip = 1)]
    pub fn write_view_mut(
        device: &mut Device,
        input: &HbmTensor<bf16, m![1], m![S]>,
        output: HbmTensorViewMut<'_, bf16, m![1], m![S]>,
    ) {
        let dm: DmTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![S]> = input.to_dm(&mut device.tdma);
        dm.view().to_hbm_view(&mut device.tdma, output);
    }
}
