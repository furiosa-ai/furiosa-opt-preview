//! Rejection fixtures for the vector engine: an ALU the chain asks for twice, an interleaved group
//! that outgrows the VE register-file cache, a VRF operand the indexer cannot address, incompatible
//! partitions, and overwriting a borrowed VRF. The legal operand shapes are
//! [`vrf_operand`](crate::vrf_operand).
//!
//! The partition fixtures are doctests rather than kernels, for the reason
//! [`negative`](crate::negative) gives: rustc refuses each of them. The first is an operand that
//! simply is not the stream's, which the operand impls reject by naming one shared set of `Chip` /
//! `Cluster` / `Slice` for both sides.
//!
//! ```compile_fail,E0277
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![W = 256, B = 32];
//!
//! // The operand is replicated over the 256 slices under an anonymous broadcast, the stream is
//! // partitioned by `W`. Same 256 slices, but the two mappings are not the same type.
//! fn add<'l, const T: Tu>(
//!     stream: BeginTensor<'l, T, i32, m![1], m![1 # 2], m![W], m![1], m![B]>,
//!     operand: VrfTensor<i32, m![1], m![1 # 2], m![256], m![B]>,
//! ) -> DmTensor<i32, m![1], m![1 # 2], m![W], m![B]> {
//!     stream
//!         .fetch::<m![B / 8], m![B % 8]>()
//!         .collect::<m![B / 8], m![B % 8]>()
//!         .vector_init()
//!         .vector_intra_slice_tag(TagMode::Zero)
//!         .vector_fxp(FxpBinaryOp::AddFxp, &operand)
//!         .vector_final()
//!         .commit_trim::<m![B % 8]>()
//!         .commit()
//! }
//! ```
//!
//! Restating the operand under the stream's slice mapping with [`VrfTensor::reshape`] is what makes
//! the same read legal. The two examples differ in that one line, so the one above fails for the
//! partition and nothing else.
//!
//! ```
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![W = 256, B = 32];
//!
//! fn add<'l, const T: Tu>(
//!     stream: BeginTensor<'l, T, i32, m![1], m![1 # 2], m![W], m![1], m![B]>,
//!     operand: VrfTensor<i32, m![1], m![1 # 2], m![256], m![B]>,
//! ) -> DmTensor<i32, m![1], m![1 # 2], m![W], m![B]> {
//!     let operand: VrfTensor<i32, m![1], m![1 # 2], m![W], m![B]> = unsafe { operand.reshape() };
//!     stream
//!         .fetch::<m![B / 8], m![B % 8]>()
//!         .collect::<m![B / 8], m![B % 8]>()
//!         .vector_init()
//!         .vector_intra_slice_tag(TagMode::Zero)
//!         .vector_fxp(FxpBinaryOp::AddFxp, &operand)
//!         .vector_final()
//!         .commit_trim::<m![B % 8]>()
//!         .commit()
//! }
//! ```
//!
//! `reshape` is not a way around the rule, which the following fixture is about: its guard is per-level
//! `SIZE` equality, so an operand spanning a different number of slices is refused before it can
//! reach an op. The guard is a `const` block, so the example calls what it declares.
//!
//! ```compile_fail
//! # #![feature(adt_const_params)]
//! # extern crate furiosa_opt_std;
//! use furiosa_opt_std::prelude::*;
//!
//! axes![W = 256, B = 32];
//!
//! fn store<'l>(
//!     input: BeginTensor<'l, { Tu::Sub }, i32, m![1], m![1 # 2], m![1 # 128], m![1], m![B]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 128], m![B]> {
//!     input.fetch::<m![1], m![B]>().collect::<m![B / 8], m![B % 8]>().to_vrf()
//! }
//!
//! // 128 slices restated as 256: `Slice::SIZE` differs, so `assert_slice_preserved` fires.
//! fn relabel(
//!     operand: VrfTensor<i32, m![1], m![1 # 2], m![1 # 128], m![B]>,
//! ) -> VrfTensor<i32, m![1], m![1 # 2], m![W], m![B]> {
//!     unsafe { operand.reshape() }
//! }
//! #
//! # fn main() -> Result<(), Error> {
//! # let mut device: Device = Device::new(Topology { chips: 1, pes: 8 })?;
//! # let input: BeginTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 128], m![1], m![B]> =
//! #     BeginTensor::new(&mut device.sub, Tensor::zero());
//! # let _ = relabel(store(input));
//! #
//! # Ok(())
//! # }
//! ```
//!
//! A VRF read view keeps its source borrowed until the consuming pipeline finishes.
//! Writing that VRF while the pipeline is pending is rejected by Rust.
//!
//! ```compile_fail,E0502
//! # #![feature(adt_const_params)]
//! use furiosa_opt_std::prelude::*;
//! # axes![S = 64, T = 2, E = 8];
//! # type Dm = DmTensor<i32, m![1], m![1], m![S], m![T, E]>;
//! # type Vrf = VrfTensor<i32, m![1], m![1], m![S], m![T, E]>;
//!
//! fn overwrite_with_pending_pipeline(device: &mut Device, source: &Dm, stored: &mut Vrf) -> Dm {
//!     let pipeline = device.main.begin(source.view())
//!         .fetch::<m![T], m![E]>().collect::<m![T], m![E]>()
//!         .vector_init().vector_intra_slice_tag(TagMode::Zero)
//!         .vector_fxp(FxpBinaryOp::AddFxp, stored.view());
//!
//!     device.sub.begin(source.view())
//!         .fetch::<m![T], m![E]>().collect::<m![T], m![E]>()
//!         .to_vrf_view(stored.view_mut()); // The pipeline still borrows `stored`.
//!
//!     pipeline.vector_final().commit_trim::<m![E]>().commit()
//! }
//! ```

use furiosa_opt_std::prelude::*;
// Bit requirements read best unqualified inside a pattern literal.
use furiosa_opt_std::prelude::BitReq::{Ignore, One, Zero};

pub use crate::vector_engine::{A, I};

axes![G = 8, D = 8, NP = 256, NA = 4, NT = 8];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

/// Three fxp ops on one chain, two of which want the same ALU: `AddFxp` takes FxpAdd, `MulInt` takes
/// FxpMul, then `SubFxp` asks for FxpAdd again.
///
/// Expected: `6 is not available for op Binary(SubFxp)`, pinned by the snapshot and by the
/// answer-key test that catches the Cpu panic.
#[device(chip = 1)]
pub fn ve_elementwise_fxp_chain(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut device.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A / 2], m![A % 2]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 10)
        .vector_fxp(FxpBinaryOp::MulInt, 2)
        .vector_fxp(FxpBinaryOp::SubFxp, 5)
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// A VRF operand whose packet lanes are `D` elements apart: the stream's packet is `G`, which the
/// operand stores a row apart, while the indexer can only read one broadcast address or a contiguous
/// run per access.
///
/// Expected: `cannot feed the 8-element packet`, pinned by the snapshot and by the answer-key test
/// that catches the Cpu panic. Storing the operand as `[D, G]` (`G` innermost) is the fix.
#[device(chip = 1)]
pub fn ve_vrf_strided_packet(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![D, G]>,
    operand: &HbmTensor<i32, Chip, m![G, D]>,
) -> HbmTensor<i32, Chip, m![D, G]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![D, G]>(&mut device.tdma);
    let operand_dm = operand.to_dm::<Cluster, Slice, m![G, D]>(&mut device.tdma);

    let operand_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![G, D]> = device
        .sub
        .begin(operand_dm.view())
        .fetch::<m![G, D / 8], m![D % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .to_vrf();

    let result: DmTensor<i32, Chip, Cluster, Slice, m![D, G]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![D], m![G]>()
        .fetch_cast::<i32>()
        .collect::<m![D], m![G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, &operand_vrf)
        .vector_final()
        .commit_trim::<m![G]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// The group axis outermost in the FETCH time, so one interleaved group is the whole 512-element row,
/// over what the VE register-file cache holds (1 KiB = 256 `i32`).
///
/// Expected: translation reports `512 elements sit inside the group axis I`, which the snapshot pins.
/// The fix is [`ve_group_pair_add_group_axis_inner`](crate::vector_engine::ve_group_pair_add_group_axis_inner).
#[device(chip = 1)]
pub fn ve_group_pair_over_cache(
    device: &mut Device,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, Slice, m![A]>(&mut device.tdma);

    let result: DmTensor<i32, Chip, Cluster, Slice, m![A]> = device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(lhs_dm.view(), rhs_dm.view())
        .fetch::<m![I, A / 8], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![I, A / 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![A / 8]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// Multi-axis VRF layouts that cannot feed a single 8-element access.
pub mod multi_axis {
    use furiosa_opt_std::prelude::*;

    axes![S = 64, T = 16, A = 2, B = 2, C = 2];

    type Chip = m![1];
    type Cluster = m![1];
    type Slice = m![S];

    /// Rejects reversed packet axes in an 8-element VRF read.
    #[device(chip = 1, pe = 1)]
    pub fn way8_axes_reversed(
        device: &mut Device,
        input: &HbmTensor<i32, Chip, m![S, T, A, B, C]>,
        bias: &HbmTensor<i32, Chip, m![S, T, C, B, A]>,
    ) -> HbmTensor<i32, Chip, m![S, T, A, B, C]> {
        let input_dm = input.to_dm::<Cluster, Slice, m![T, A, B, C]>(&mut device.tdma);
        let bias_dm = bias.to_dm::<Cluster, Slice, m![T, C, B, A]>(&mut device.tdma);

        let bias_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![T, C, B, A]> = device
            .sub
            .begin(bias_dm.view())
            .fetch::<m![T], m![C, B, A]>()
            .collect::<m![T], m![C, B, A]>()
            .to_vrf();

        device
            .main
            .begin(input_dm.view())
            .fetch::<m![T], m![A, B, C]>()
            .collect::<m![T], m![A, B, C]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &bias_vrf)
            .vector_final()
            .commit_trim::<m![A, B, C]>()
            .commit::<m![T, A, B, C]>()
            .to_hbm(&mut device.tdma)
    }

    /// Rejects a partial packet broadcast in an 8-element VRF read.
    #[device(chip = 1, pe = 1)]
    pub fn way8_axes_partial_broadcast(
        device: &mut Device,
        input: &HbmTensor<i32, Chip, m![S, T, A, B, C]>,
        bias: &HbmTensor<i32, Chip, m![S, T, A, C]>,
    ) -> HbmTensor<i32, Chip, m![S, T, A, B, C]> {
        let input_dm = input.to_dm::<Cluster, Slice, m![T, A, B, C]>(&mut device.tdma);
        let bias_dm = bias.to_dm::<Cluster, Slice, m![T, A, C]>(&mut device.tdma);

        let bias_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![T, A, C]> = device
            .sub
            .begin(bias_dm.view())
            .fetch::<m![T / 2], m![T % 2, A, C]>()
            .collect::<m![T / 2], m![T % 2, A, C]>()
            .to_vrf();

        device
            .main
            .begin(input_dm.view())
            .fetch::<m![T], m![A, B, C]>()
            .collect::<m![T], m![A, B, C]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &bias_vrf)
            .vector_final()
            .commit_trim::<m![A, B, C]>()
            .commit::<m![T, A, B, C]>()
            .to_hbm(&mut device.tdma)
    }

    /// Rejects packet lanes split into two runs by an intervening `T` axis.
    #[device(chip = 1, pe = 1)]
    pub fn way8_axes_two_runs(
        device: &mut Device,
        input: &HbmTensor<i32, Chip, m![S, T, A, B, C]>,
        bias: &HbmTensor<i32, Chip, m![S, A, T, B, C]>,
    ) -> HbmTensor<i32, Chip, m![S, T, A, B, C]> {
        let input_dm = input.to_dm::<Cluster, Slice, m![T, A, B, C]>(&mut device.tdma);
        let bias_dm = bias.to_dm::<Cluster, Slice, m![A, T, B, C]>(&mut device.tdma);

        let bias_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![A, T, B, C]> = device
            .sub
            .begin(bias_dm.view())
            .fetch::<m![A, T / 2], m![T % 2, B, C]>()
            .collect::<m![A, T / 2], m![T % 2, B, C]>()
            .to_vrf();

        device
            .main
            .begin(input_dm.view())
            .fetch::<m![T], m![A, B, C]>()
            .collect::<m![T], m![A, B, C]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &bias_vrf)
            .vector_final()
            .commit_trim::<m![A, B, C]>()
            .commit::<m![T, A, B, C]>()
            .to_hbm(&mut device.tdma)
    }
}

/// Three immediate branches and an RF branch reading the stash, where the RF branch adds a value no
/// immediate branch declares. The RF branch reads the first immediate register, so a value of its
/// own takes that register and leaves two for the three immediate branches.
///
/// Expected: `has to add what the first immediate branch adds`, pinned by the snapshot. Declaring
/// that value on the first immediate branch is the fix: the two branches then read one register and
/// all four stand.
#[device(chip = 1)]
pub fn ve_ternary_stash_over_immediate_budget(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, m![A / 2], m![A % 2]>(&mut device.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A / 2], m![A % 2]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 2]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![A % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Equal(0.0),
            Cmp::Greater(0.0),
            Cmp::Less(0.0),
            Cmp::Greater(100.0),
        ]))
        .vector_narrow_trim::<m![A % 2 # 4]>()
        .vector_stash()
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(TagGuard::matches([One, Ignore, Ignore, Ignore]), (0.0f32, 7.0f32))
                .imm(TagGuard::matches([Zero, Zero, One, Ignore]), (-3.0f32, 100.0f32))
                .imm(TagGuard::matches([Zero, One, Ignore, One]), (5.0f32, 50.0f32))
                .rf(TagGuard::matches([Zero, One, Ignore, Zero]), (Stash, 1000.0f32)),
        )
        .vector_widen_pad::<m![A % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![A % 2]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}

/// The [`ve_ternary_stash_over_immediate_budget`] layout with a VRF read in place of the stash, so
/// the rejection is pinned for both ports the rf slot can name.
#[device(chip = 1)]
pub fn ve_ternary_vrf_over_immediate_budget(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![NP, NA, NT]>,
    scale: &HbmTensor<f32, Chip, m![NP, NT]>,
) -> HbmTensor<f32, Chip, m![NP, NA, NT]> {
    let input_dm = input.to_dm::<Cluster, m![NP], m![NA, NT]>(&mut device.tdma);
    let scale_dm = scale.to_dm::<Cluster, m![NP], m![NT]>(&mut device.tdma);

    let scale_vrf: VrfTensor<f32, Chip, Cluster, m![NP], m![NT]> = device
        .sub
        .begin(scale_dm.view())
        .fetch::<m![1], m![NT]>()
        .collect::<m![1], m![NT]>()
        .to_vrf();

    let result: DmTensor<f32, Chip, Cluster, m![NP], m![NA, NT]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![NA], m![NT]>()
        .collect::<m![NA], m![NT]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Equal(0.0),
            Cmp::Greater(0.0),
            Cmp::Less(0.0),
            Cmp::Greater(100.0),
        ]))
        .vector_narrow_split::<m![NA, NT / 4], m![NT % 4]>()
        .vector_fp_ternary(
            FpTernaryOp::FmaF,
            Branched::imm(TagGuard::matches([One, Ignore, Ignore, Ignore]), (0.0f32, 7.0f32))
                .imm(TagGuard::matches([Zero, Zero, One, Ignore]), (-3.0f32, 100.0f32))
                .imm(TagGuard::matches([Zero, One, Ignore, One]), (5.0f32, 50.0f32))
                .rf(TagGuard::matches([Zero, One, Ignore, Zero]), (&scale_vrf, 1000.0f32)),
        )
        .vector_widen_concat::<m![NA], m![NT]>()
        .vector_final()
        .commit_trim::<m![NT]>()
        .commit();

    result.to_hbm(&mut device.tdma)
}
