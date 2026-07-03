use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32, D = 64];

/// Device function that transposes a tensor from shape [A, B] to [B, A].
#[device(chip = 1)]
pub fn tile_simple(ctx: &mut Context, input: HbmTensorView<'_, i8, m![1], m![A, B]>) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, A]>::from_addr(0x3000) };
    for b in 0..32 {
        // TODO: replace with <m![B]>::SIZE
        let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}

type Chip = m![1];
type Cluster = m![1];
type Slice = m![1 # 256];

/// Commits a fetched 32-wide window into the *upper* half of a 64-wide DM via `commit_view` into a
/// `view_mut().tile()`. The tile's out-of-tile cells are down (Bottom) padding, so the commit
/// sequencer places the `/ 8` strided time within the live prefix and leaves them unwritten rather
/// than rejecting the write. Copies `input[0..32]` into `result[32..64]`. Regression for a
/// `view_mut().tile()` commit into a windowed-and-down-padded DM that surfaced as
/// `StreamUnmatchedSegment`.
#[device(chip = 1)]
pub fn tile_window_commit(ctx: &mut Context, input: &HbmTensor<f32, Chip, m![D]>) -> HbmTensor<f32, Chip, m![D]> {
    let tensor: DmTensor<f32, Chip, Cluster, Slice, m![D]> = input.to_dm(&mut ctx.tdma);
    let tile_one = tensor.view().tile::<m![D], 32, m![D = 32 # 64]>(0);

    let mut result: DmTensor<f32, Chip, Cluster, Slice, m![D]> = unsafe { DmTensor::from_addr(1 << 12) };

    ctx.main
        .begin(tile_one)
        .fetch::<m![1], m![D = 32]>()
        .fetch_cast::<f32>()
        .collect::<m![D = 32 / 8], m![D = 32 % 8]>()
        .commit_trim::<m![D = 32 % 8]>()
        .commit_view(result.view_mut().tile::<m![D], 32, m![D = 32 #{!} 64]>(32));

    result.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Same transpose as [`tile_simple`], but the input is aliased with a bare `view()`
/// inside the loop before tiling. The bare view resolves directly to the outer
/// param tensor, which the loop-body builder must find among the parent's tensors;
/// without that it fails VISA -> LIR with "no tensor exists for T{n}".
#[device(chip = 1)]
pub fn tile_view_in_loop(ctx: &mut Context, input: &HbmTensor<i8, m![1], m![A, B]>) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, A]>::from_addr(0x3000) };
    for b in 0..32 {
        let input_slice = input.view().tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}

/// Identity copy through a two-level HBM-tiled loop nest whose *inner* body tiles a view
/// produced by the *outer* loop variable -- the loop-body capture shape of the attention-decode
/// kernels. The outer `for g` slices `input` into a 16-wide `B`-group (`in_group`, tiled by the
/// outer index `g`); the inner `for h` tiles *that group* by the inner index, so the inner loop
/// body reads a tensor whose `IndexAccess` `table` traces back through `in_group` to the OUTER
/// loop's index tensor -- an index allocated in the enclosing scope, absent from the inner body's
/// own tensors.
///
/// An HBM-tiled loop is not unrolled (contrast the SRAM nest in [`tile_computed_offset`]), so both
/// `for`s survive as kept VISA `Loop`s. `convert_loop` must therefore (1) thread the outer index
/// into the inner loop's `captured_inputs` -- else VISA -> LIR hits the loop-body-graph verify wall
/// (`Loop::build_body_graph -> verify_structural_validity`, "no tensor exists for T{n}") -- and
/// (2) keep the inner nest's own locals OUT of the outer loop's `local_tensors` -- else the LIR
/// loop-body completeness check rejects them ("contained in local_tensors but not referenced by any
/// local instructions"). It is the in-repo regression for both facets of that fix; the matching
/// `test_tile_view_in_nested_loop` is a lowering-only check (`build_lir`, not `compare_lir!`) because
/// the Simulation backend's tile-shape validator does not accept this two-level HBM tile.
#[device(chip = 1)]
pub fn tile_view_in_nested_loop(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![A, B]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![A, B]>::from_addr(0x3000) };
    for g in 0..2 {
        // Outer-index tile: a 16-wide `B`-group selected by the enclosing loop var `g`. Its
        // `tile` takes `&self`, so the inner loop can re-slice it every iteration; the index
        // operand it carries (`g`) is allocated in the enclosing loop scope.
        let in_group = input.view().tile::<m![B / 16], 1, m![A, 1 # 2, B % 16]>(g);
        for h in 0..16 {
            // Inner-index tile of the outer-index view: the read tensor's index operand (`h`)
            // is inner-local, but its `table` traces back through `in_group` to the outer `g`.
            let in_col = in_group.tile::<m![B % 16], 1, m![A, 1 # 32]>(h);
            // The write side re-slices `output` by the same outer group index `g` and inner
            // column index `h`; the outer `g` view is rebuilt here (the mutable `tile` consumes
            // its receiver) but still references the enclosing `g` index tensor.
            let out_col = output
                .view_mut()
                .tile::<m![B / 16], 1, m![A, 1 # 2, B % 16]>(g)
                .tile::<m![B % 16], 1, m![A, 1 #{!} 32]>(h);
            in_col.to_hbm_view(&mut ctx.tdma, out_col);
        }
    }
    output
}

/// Chunked-output write created *inside* a loop and returned *after* it: the exact
/// T1 pattern that VISA -> LIR lowering must keep visible past the loop op.
///
/// Copies `input[B, D]` to `output[B, D]` one `D`-chunk per iteration. Each
/// `output.view_mut().tile::<m![D / 16], 1, m![B, 1 #{!} 4, D % 16]>(i)` is a
/// chunked write tile whose strided `D_16` chunk-index axis is absent from the
/// contiguous output shape, so lowering must materialize it (`split_chunk_index_axis`).
/// The write tile is a loop-body-local tensor whose writeback resolves against the
/// returned `output` *after* the loop closes; dropping loop-body locals from the parent
/// regresses this to a "no tensor T{n}" panic. Together this is a single runnable value
/// oracle (`compare_lir!`) for both the loop-tensor re-merge and the chunk-axis fix.
#[device(chip = 1)]
pub fn tile_chunked_output(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![B, D]>,
) -> HbmTensor<i8, m![1], m![B, D]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, D]>::from_addr(0x3000) };
    for i in 0..4 {
        let input_slice = input.view().tile::<m![D / 16], 1, m![B, 1 # 4, D % 16]>(i);
        let output_slice = output.view_mut().tile::<m![D / 16], 1, m![B, 1 #{!} 4, D % 16]>(i);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}
