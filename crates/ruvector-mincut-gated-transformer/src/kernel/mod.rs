//! Kernel operations for quantized inference.
//!
//! This module provides the core mathematical operations:
//! - Quantized GEMM (int8 matrix multiplication)
//! - Layer normalization
//! - Activation functions
//! - Benchmark utilities

pub mod qgemm;
pub mod norm;
pub mod bench_utils;

pub use qgemm::{qgemm_i8, qgemm_i8_simd};
pub use norm::{layer_norm, layer_norm_inplace, rms_norm};
pub use bench_utils::{Timer, BenchStats, BenchConfig, run_benchmark, compute_gflops, compute_bandwidth_gbps};
