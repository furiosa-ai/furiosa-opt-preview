// Arguments: output followed by values. Each cluster reports sum, last, count and column, where
// the column is chip * 2 + cluster: two clusters on each chip.
// Device code for tests/device.rs; `make furiosa-opt-fixtures` assembles the .bin beside it.
// 0xf_ffxx addresses are fields of libpe's `shared_field_t` (libpe-artifacts/src/libpe/tuc.h).
.text
.global entry
entry:
    mov x10, x1
    ldr x4, [x0], #8
    mov x5, #0xc000000000  // DRAM_BASE
    add x4, x4, x5
    mov x5, #0xffe8
    movk x5, #0xf, lsl #16  // shared->pe_id
    ldr x6, [x5]
    lsr x6, x6, #2  // cluster on the chip: four PEs each
    add x4, x4, x6, lsl #5  // 32 bytes of record per cluster
    ldur x7, [x5, #-8]  // shared->npu_vid
    add x7, x6, x7, lsl #1
    mov x8, #0
    mov x9, #0
    subs x1, x1, #1
    b.eq 2f
1:
    ldr x9, [x0], #8
    add x8, x8, x9
    subs x1, x1, #1
    b.ne 1b
2:
    stp x8, x9, [x4]
    stp x10, x7, [x4, #16]
    dsb sy
    mov x3, #0xffc0
    movk x3, #0xf, lsl #16  // shared->trampoline
    ldr x3, [x3]
    mov x0, #0
    mov x1, #0
    mov x2, #0
    br x3
