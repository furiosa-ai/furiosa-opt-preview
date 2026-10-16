# furiosa-opt-abi

The device ABI: the bytes and registers the compiler, the host runtime, the firmware and the
signed bootloader agree on. Applications use `furiosa-opt-std`; this crate is internal to it, to
[`furiosa-opt-rt`](../furiosa-opt-rt/README.md) and to the device firmware.

`cargo furiosa-opt` compiles each `#[device]` function and links the result into the application.
At run time `furiosa-opt-rt` boots the firmware through the signed bootloader, then sends it IPC
frames on a ring and doorbell; a launch frame names an FOE1 executable, its memory `Kind`s and
its `Topology`. The firmware enters the executable's chunks through the task ABI and serves their
yields.

This crate owns every contract that reaches the device: the bootloader exchange, the ring and
doorbell, the executable format and the memory and topology types. The IPC frames travel on its
rings but are versioned by `furiosa-opt-ipc`. The task ABI is local to the firmware. The
compiler's image, described next, never reaches the device.

## Compiler output

The compiler hands an application its device functions as native code. Its link pass compiles a
generated crate into each executable, which carries:

- a registry: one `furiosa_opt_rt::registry::Entry { path, key, image }` per launched instantiation,
  in the `furiosa_kernels` link section, sorted by path then key. `furiosa-opt-std` reads the
  section between its `__start_` and `__stop_` symbols and binary-searches it for the function a
  launch names. `furiosa_opt_rt::registry` owns the entry layout.
- per entry, a loader of type `furiosa_opt_rt::registry::Image`, which turns a device into a loaded
  `Function`. This is all `furiosa-opt-std` and the runtime see of a compiled function.
- per entry, an image: the FOE1 executable defined here, and the compiler's statics, argument
  slots and profile spans in a format private to the compiler. Only the loader reads it.

Inside a loader, compiler-generated host code uses a second, compiler-private interface of
`furiosa-opt-rt`: `unsafe Function::load` stages the FOE1 bytes and takes a bind callback that
maps a caller's arguments to the operand words device code reads
(`Arguments::bind`) and returns the future that completes the launch. A change to that
interface touches only the compiler and its generated code; a change to `Entry` or `Image` also
touches `furiosa-opt-std`. Both sides come from one SDK build, since a program loads only with
the SDK, toolchain and target it was built with.

## What device code receives

A chunk's entry takes opaque machine words. The task ABI fixes only how an entry is entered (x0
holds the words' address, x1 their count) and how it yields. What the words mean (tensor
bindings, returns, allocation plans, profile capture) is written by the bind callback and read by
compiler-generated device code. Changing their meaning therefore changes the image format, never
the device ABI.

## Contracts

1. `executable` (FOE1): magic, table offset, entry stride, chunk count, cluster count; entries
   chunk-major, each a code range and an entry offset; code 256-byte aligned after the table,
   entry offset 4-byte aligned inside its code, at most `CHUNK_LIMIT` bytes per chunk.
2. `bootloader`: `Request { addr, len, token }`; the firmware lands at `FIRMWARE_BASE` and opens
   with `magic(token)`; the answer is one `Completion` whose first payload word is a `Code`.
3. `ring`: `u64` entries and two `u32` index registers, one slot always empty; every entry opens
   with a `header` `{ id, words }`; a `Completion` is a header and fixed payload words.
4. `doorbell`: a status word whose bits name the senders with pending entries, and one ring per
   sender slot.
5. `memory::Kind`: `Dram`, `Sram`, `DramShared`.
6. `Topology { chips, pes }`: the device a compiled program targets.
7. `arch::rngd`: RNGD's numbers: device-memory window, doorbell and completion layout, SRAM
   window, DMA operand encoding and configuration, chip limits.

The `alloc` feature, on by default, enables executable encoding and DMA segment tables, and
`serde` derives serialization for `memory::Kind` and `Topology`.
