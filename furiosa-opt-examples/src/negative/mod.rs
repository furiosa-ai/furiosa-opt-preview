//! Kernels that MUST fail to compile: each one states something the language or the hardware does
//! not allow, and exists so the compiler's rejection stays pinned.
//!
//! Most of these kernels still run on the CPU, because what they violate is a
//! device-translation rule rather than an arithmetic one. The answer-key tests exercise that,
//! asserting the panic where the emulator catches it too; only compilation must fail.
//!
//! The rejections rustc itself enforces -- the VRF operand partition rules in [`vector_engine`],
//! the per-PE SPM budget in [`scatter_gather`] -- cannot be kernels here at all: stating one would
//! stop this crate compiling. They live as `compile_fail` doctests in those modules' docs, so
//! `cargo test --doc` is their whole gate.

pub mod blocking_flits;
pub mod contract_outer_assertions;
pub mod dm_slice;
pub mod dma;
pub mod fetch_lift;
pub mod generic_device;
pub mod memset;
pub mod runtime_if_scalar;
pub mod runtime_panic;
pub mod scalar_cast_diag;
pub mod scatter_gather;
pub mod switch_assertions;
pub mod transpose;
pub mod unroll_loop;
pub mod vector_engine;
