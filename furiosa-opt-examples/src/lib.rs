//! Virtual ISA programs.
//!
//! Every concrete kernel outside [`negative`] compiles end-to-end to an EDF; [`generic`]'s
//! fns compile from a linked root's launches instead, and [`negative`] holds the rejection
//! fixtures, which must fail. Legal programs the compiler cannot lower yet are not here
//! at all: they live in the private `npu-opt-examples` crate under `unsupported`.

#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod algebraic;
pub mod always_panics;
pub mod bias_partition_broadcast;
pub mod binary_add;
pub mod chip_reduce;
pub mod cluster_chip_shuffle_slice;
pub mod cluster_reduce;
pub mod commit_view_tile;
pub mod contract_element_types;
pub mod contract_outer_assertions;
pub mod dma;
pub mod dma_halves;
pub mod dma_wave;
pub mod double_buffering;
pub mod dynamic_extent;
pub mod exclusive_arms;
pub mod extent_accumulate;
pub mod extent_stream;
pub mod f4_indexing;
pub mod fetch_assertions;
pub mod fetch_commit;
pub mod fetch_lift;
pub mod fetch_table_lookup;
pub mod generic;
pub mod host_tile_view;
pub mod layer_pool;
pub mod live_window_transpose;
pub mod matmul;
pub mod memory_op;
pub mod memset;
pub mod mnist;
pub mod moe;
pub mod negative;
pub mod nested_extent;
pub mod padded_axis_tile;
pub mod param;
pub mod pe_count;
pub mod prefix_lengths_loop;
pub mod reduce_span;
pub mod reshape;
pub mod root_view;
pub mod runtime_dm;
pub mod runtime_if;
pub mod runtime_if_scalar;
pub mod runtime_rows;
pub mod runtime_rows_fill;
pub mod runtime_rows_fold;
pub mod runtime_rows_norm;
pub mod runtime_rows_reduce;
pub mod runtime_rows_scaled;
pub mod runtime_rows_silicon;
pub mod runtime_rows_spread;
pub mod runtime_rows_tiles;
pub mod runtime_rows_trf;
pub mod runtime_rows_vector;
pub mod runtime_rows_vector_silicon;
pub mod runtime_rows_window;
pub mod scalar;
pub mod scatter_gather;
pub mod scatter_gather_topology;
pub mod slice_span;
pub mod stotrf_table_lookup;
pub mod switch_assertions;
pub mod switch_bitmap_loop;
pub mod switch_engine;
pub mod tile;
pub mod to_vrf_assertions;
pub mod transformer;
pub mod transpose;
pub mod typelevel_const;
pub mod unevaluated_const;
pub mod unroll_loop;
pub mod vector_engine;
pub mod view;
pub mod vrf_add_segmented;
pub mod vrf_operand;
