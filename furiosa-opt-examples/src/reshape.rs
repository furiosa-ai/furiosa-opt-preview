#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 256, B = 4096, C = 4, D = 2, E = 16, F = 16, G = 16, H = 2, I = 16,];

#[device(chip = 4)]
pub fn reshape(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![C], m![D, E, F, G, H, I]> {
    let hbm_transposed: HbmTensor<i32, m![A / 4 % 4], m![A % 4, B, A / 16]> = hbm_tensor.to_hbm(&mut device.tdma);
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_transposed.to_dm(&mut device.tdma);

    let reshaped: DmTensor<i32, m![C], m![D], m![E, F], m![G, H, I]> = unsafe { dm_tensor.reshape() };

    reshaped.to_hbm(&mut device.tdma)
}

// Note: This test uses different shape definitions, so it's in a separate module
pub mod different_axes {
    use furiosa_opt_std::prelude::*;

    axes![A = 256, B = 4096, C = 4, D = 2, E = 256, F = 512,];

    #[device(chip = 4)]
    pub fn reshape_different_num_axes(
        device: &mut Device,
        hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
    ) -> HbmTensor<i32, m![C], m![D, E, F]> {
        let hbm_transposed: HbmTensor<i32, m![A / 4 % 4], m![A % 4, B, A / 16]> = hbm_tensor.to_hbm(&mut device.tdma);

        let dm_tensor: DmTensor<
            i32,
            m![A / 4 % 4],
            m![A / 2 % 2],
            m![B % 16, B / 16 % 16],
            m![B / 256, A % 2, A / 16],
        > = hbm_transposed.to_dm(&mut device.tdma);

        let reshaped: DmTensor<i32, m![C], m![D], m![E], m![F]> = unsafe { dm_tensor.reshape() };

        reshaped.to_hbm(&mut device.tdma)
    }
}

pub mod hbm_view {
    use furiosa_opt_std::prelude::*;

    axes![Rows = 128, Tile = 64, Width = 16];

    /// Copies the last tile into the first through reshaped read and write views.
    #[device(chip = 1)]
    pub fn copy_tile(
        device: &mut Device,
        input: &HbmTensor<i32, m![1], m![Rows, Width]>,
        output: &mut HbmTensor<i32, m![1], m![Rows, Width]>,
    ) {
        // SAFETY: both reshapes retain the selected tile's wire order and outer stride.
        let input = unsafe {
            input
                .view()
                .tile::<m![Rows / Tile], 1, m![Rows / Tile = 1 # 2, Rows % Tile, Width]>(1)
                .reshape::<m![1], m![1 # 2, Tile, Width]>()
        };
        let output = unsafe {
            output
                .view_mut()
                .tile::<m![Rows / Tile], 1, m![Rows / Tile = 1 #{!} 2, Rows % Tile, Width]>(0)
                .reshape::<m![1], m![1 #{!} 2, Tile, Width]>()
        };
        input.to_hbm_view(&mut device.tdma, output);
    }
}
