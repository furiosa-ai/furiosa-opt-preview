// Arguments: source, destination, bytes/cluster, transfers/cluster.
// Each cluster copies its range to the next rank in disjoint transfers; two use both owned channels.
// Device code for tests/device.rs; `make furiosa-opt-fixtures` assembles the .bin beside it.
// 0xf_ffxx addresses are fields of libpe's `shared_field_t` (libpe-artifacts/src/libpe/tuc.h).
.text
.global entry
entry:
    mov x19, x0
    mov x21, #0xffe8
    movk x21, #0xf, lsl #16  // shared->pe_id
    ldr x21, [x21]
    lsr x21, x21, #2  // cluster on the chip: four PEs each
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
    sub sp, sp, #64
    mov x3, #0xff80
    movk x3, #0xf, lsl #16  // shared->num_peers
    ldr x3, [x3]
    add x28, x28, #1
    cmp x28, x3
    csel x28, xzr, x28, eq
    mov x4, #16
    sub x4, x4, x3
    udiv x4, x4, x3
    madd x28, x28, x4, x3
    lsl x28, x28, #32
    mov x3, #0x8000000000  // REMOTE_PE_DATA_BASE: the next rank's memory
    add x28, x28, x3
    mov x27, #0
2:
    ldr x3, [x19, #24]
    udiv x2, x22, x3
    mul x4, x27, x2
    ldr x0, [x19]
    mov x3, #0xc000000000  // DRAM_BASE
    add x0, x0, x3
    add x0, x0, x23
    add x0, x0, x4
    ldr x1, [x19, #8]
    add x1, x1, x28
    add x1, x1, x23
    add x1, x1, x4
    blr x24
    strb w0, [sp, x27]
    add x27, x27, #1
    ldr x3, [x19, #24]
    cmp x27, x3
    b.ne 2b
    mov x27, #0
3:
    ldrb w0, [sp, x27]
    orr x0, x0, #0x300  // TRAMPOLINE_COMPLETE_DMA | id: wait for the transfer
    mov x1, #0
    mov x2, #0
    blr x25
    add x27, x27, #1
    ldr x3, [x19, #24]
    cmp x27, x3
    b.ne 3b
    add sp, sp, #64
    mov x0, #0
    mov x1, #0
    mov x2, #0
    br x25  // trampoline(TRAMPOLINE_EXIT, 0, 0)
