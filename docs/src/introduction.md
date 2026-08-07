# Introduction

FuriosaAI's Tensor Contraction Processor (TCP) is a massively parallel AI accelerator targeting inference workloads.
Unlike high-level frameworks like PyTorch and XLA, which abstract away memory layouts and hardware scheduling, TCP exposes direct programmer control without requiring the byte-level reasoning of low-level kernel APIs.

TCP's Virtual Instruction Set Architecture (Virtual ISA, or vISA) is the programming interface that exposes this control.
It lets programmers reason in tensors while directly managing memory allocation and Tensor Unit scheduling.
This manual primarily helps authors guide an AI agent to write, verify, and optimize vISA kernels.
It also provides the reference material that compiler developers need when generating vISA.
[The `furiosa-opt-std` rustdoc](https://docs.rs/crate/furiosa-opt-std) is the canonical API source for published releases.
This book explains how to make and review kernel design decisions.
Both audiences assume basic Rust familiarity.
See [the language manual](https://doc.rust-lang.org/book/) if needed.

## Read This Book by Task

Start with [Quick Start](./quick-start.md) to understand a base-template kernel pattern.
Read [Mapping Tensors](./mapping-tensors/index.md), [Moving Tensors](./moving-tensors/index.md), [Computing Tensors](./computing-tensors/index.md), and [Scheduling](./scheduling.md) when a pattern needs a new capability.

> [!WARNING]
> **Alpha Test Build: Experimental Software**
>
> This software is an early, experimental, and incomplete build intended strictly for technical evaluation and internal testing.
>
> Before using this software for any production work, critical tasks, or for important data, you must consult with Furiosa engineers.
>
> Your feedback is vital to our development.
> Please provide it.

## Installation

Install three pieces:

1. **Rust toolchain (pinned)**: the Furiosa optimizer is a rustc driver, ABI-locked to a specific nightly.

   ```bash
   rustup toolchain install nightly-2026-05-01
   ```

   The same channel is pinned in [`rust-toolchain.toml`](https://github.com/furiosa-ai/furiosa-opt/blob/main/rust-toolchain.toml); cargo activates it automatically when you cd into a project that includes that file.

2. **[`cargo-furiosa-opt`](./appendix/cargo-furiosa-opt.md)**: the cargo subcommand that injects the right `--cfg backend="..."` and pre-compiles kernels for NPU.

   ```bash
   cargo +nightly-2026-05-01 install cargo-binstall
   cargo +nightly-2026-05-01 binstall cargo-furiosa-opt
   ```

3. **Furiosa SDK + physical NPU** *(only for `--backend npu`)*: the NPU backend dispatches to real hardware via the SDK's kernel driver and PE runtime (`furiosa-driver-rngd`, `furiosa-smi`, etc.; see the [SDK documentation](https://developer.furiosa.ai/latest/en/)).

   **The `emulation` and `typecheck` backends do not require the SDK.** They run host-side with no NPU dependency, so a customer who only intends to develop or evaluate kernels does not need to install the SDK at all.

## Your First Program

Use [`cargo-generate`](https://cargo-generate.github.io/cargo-generate/) to scaffold the `base-template` starter:

```bash
cargo install cargo-generate
cargo generate furiosa-ai/furiosa-opt base-template
cd base-template
```

The template is the canonical executable example and CI generates the current source under an arbitrary name before testing it with emulation.
Read [Quick Start](./quick-start.md) before adapting it, including [Use the Kernel Authoring Skill](./quick-start.md#use-the-kernel-authoring-skill) when an agent implements the change.

### Template Layout

The generated `Cargo.toml` declares `[package.metadata.furiosa-opt]`, which marks the package for kernel compilation.
`src/kernel/` contains each `#[device]` function and `src/kernel/mod.rs` re-exports those kernel modules.
Direct `src/*.rs` files are host binaries that call `launch(kernel, ...)` and contain their host-oracle tests.
Each host binary has an explicit `[[bin]]` entry with `path = "src/<name>.rs"` in `Cargo.toml`.
Do not move host binaries into `src/bin/`, `examples/`, or `tests/`, because the compiler plugin scans cargo targets rooted at `src/`.

```text
base-template/
├── Cargo.toml
├── rust-toolchain.toml
└── src/
    ├── lib.rs
    ├── kernel/
    │   ├── mod.rs
    │   └── <name>_kernel.rs
    └── <name>.rs
```

### Add a Kernel

1. Add `src/kernel/<name>_kernel.rs` with a `#[device(...)] pub fn <name>_kernel(...)` function.
2. Re-export it from `src/kernel/mod.rs`.
3. Add `src/<name>.rs` with the host program, host oracle, and a call to `launch(<name>_kernel, ...)`.
4. Register the host program as a `[[bin]]` target in `Cargo.toml`.
5. Run the kernel with `cargo furiosa-opt run --release --bin <name>`.

### Run and Verify a Template Example

```bash
# Run values with the default emulation backend.
cargo furiosa-opt run --release --bin gemm

# Execute the kernel body with phantom tensors to check mappings and shapes.
cargo furiosa-opt --backend typecheck run --release --bin gemm

# Compare emulated values with the host oracle.
cargo furiosa-opt --backend emulation test --release --bin gemm

# Run the hardware path when the SDK and an NPU are available.
cargo furiosa-opt --backend npu run --release --bin gemm
```

`typecheck test` is not numerical-correctness evidence because the phantom output can make an elementwise comparison loop execute zero iterations.

## Development Tools

The Furiosa IR Optimizer provides utilities for developing, testing, and optimizing vISA programs on TCP devices.
It complements the [Furiosa SDK's compiler](https://developer.furiosa.ai/latest/en/overview/software_stack.html#furiosa-compiler) by giving developers fine-grained control over program behavior, whether the programmer writes vISA by hand or a compiler generates it.

### Backends

A vISA program is a Rust program that uses the `furiosa-opt-std` API. `cargo furiosa-opt` selects which backend evaluates the kernel by setting `--cfg backend="..."`:

| Backend | Default? | What runs | Use when |
|---------|----------|-----------|----------|
| `typecheck` | | Kernel body runs with phantom tensors and skips value computation. | Catching mapping and shape errors quickly. |
| `emulation` | yes | Host-side buffer emulation of tensor operations. | Checking numerical correctness against a host oracle without an NPU. |
| `npu` | | Compiled EDF on hardware. | Running the hardware path with the Furiosa SDK and an NPU. |

`cargo check` checks types.
`cargo furiosa-opt --backend typecheck run` also executes the kernel body, so it catches additional mapping and shape errors.
`cargo check` does not execute kernel bodies, so use `typecheck run` when a mapping assertion must execute.
`cargo furiosa-opt` forwards cargo subcommands and flags, so `build`, `check`, `run`, and `test` retain their normal cargo target selection.

See the [`cargo-furiosa-opt` appendix](./appendix/cargo-furiosa-opt.md) for commands and the [rustdoc](https://docs.rs/crate/furiosa-opt-std) for API details.

### Language Server

`furiosa-rust-analyzer-proxy` provides standard Rust IDE features with readable Furiosa mapping expressions.
It renders verbose mapping types such as `Stride<Symbol<A>, 8>` as `m![A / 8]`.
See the [Language Server appendix](./appendix/language-server.md) for installation and configuration.

### Schedule Viewer

The Schedule Viewer visualizes the execution timeline to help identify performance bottlenecks.
Use `furiosa-opt` to export a schedule JSON file, then open it with `furiosa-schedule-viewer`.

For installation and usage, see the [Schedule Viewer appendix](./appendix/schedule-viewer.md).

## License

This documentation and the entire `furiosa-opt` repository are licensed under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
