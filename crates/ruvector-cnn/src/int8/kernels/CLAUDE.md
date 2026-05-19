# ruvector-cnn/src/int8/kernels

INT8 kernel implementations dispatched by `int8/mod.rs`.

## Files

- `mod.rs` — kernel module entry / arch dispatch.
- `scalar.rs` — portable scalar reference path.
- `simd.rs` — SIMD-accelerated path.

Reference path kept for cross-checks (see `tests/kernel_equivalence.rs`).
