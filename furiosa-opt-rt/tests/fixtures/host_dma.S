// Arguments: source, destination, bytes/cluster, channels/cluster, host offset, clusters/chip.
// Host operands carry the SG tag; device operands are local HBM offsets.
// Device code for tests/device.rs; `make furiosa-opt-fixtures` assembles the .bin beside it.
// 0xf_ffxx addresses are fields of libpe's `shared_field_t` (libpe-artifacts/src/libpe/tuc.h).
.text
.global entry
entry:
    mov x19, x0
    // A one-cluster device may run on pe4, so the cluster index wraps to the clusters it has.
    mov x21, #0xffe8
    movk x21, #0xf, lsl #16  // shared->pe_id
    ldr x21, [x21]
    lsr x21, x21, #2  // cluster on the chip: four PEs each
    ldr x3, [x19, #40]
    sub x4, x3, #1
    and x21, x21, x4
    ldr x22, [x19, #16]
    mul x23, x21, x22
    mov x24, #0xff90
    movk x24, #0xf, lsl #16  // shared->dma_send
    ldr x24, [x24]
    mov x25, #0xffc0
    movk x25, #0xf, lsl #16  // shared->trampoline
    ldr x25, [x25]
    mov x28, #0xffe0
    movk x28, #0xf, lsl #16  // shared->npu_vid
    ldr x28, [x28]
    madd x28, x28, x3, x21
    ldr x3, [x19, #32]
    madd x28, x28, x22, x3
    sub sp, sp, #16
    mov x27, #0
2:
    ldr x3, [x19, #24]
    udiv x2, x22, x3
    mul x4, x27, x2
    mov x3, #0xc000000000  // DRAM_BASE
    ldr x0, [x19]
    tbnz x0, #63, 3f  // SG tag: a host operand
    add x0, x0, x3
    add x0, x0, x23
    b 4f
3:
    add x0, x0, x28
4:
    add x0, x0, x4
    ldr x1, [x19, #8]
    tbnz x1, #63, 5f
    add x1, x1, x3
    add x1, x1, x23
    b 6f
5:
    add x1, x1, x28
6:
    add x1, x1, x4
    blr x24
    strb w0, [sp, x27]
    add x27, x27, #1
    ldr x3, [x19, #24]
    cmp x27, x3
    b.ne 2b
    mov x27, #0
7:
    ldrb w0, [sp, x27]
    orr x0, x0, #0x300  // TRAMPOLINE_COMPLETE_DMA | id: wait for the transfer
    mov x1, #0
    mov x2, #0
    blr x25
    add x27, x27, #1
    ldr x3, [x19, #24]
    cmp x27, x3
    b.ne 7b
    add sp, sp, #16
    mov x0, #0
    mov x1, #0
    mov x2, #0
    br x25  // trampoline(TRAMPOLINE_EXIT, 0, 0)
