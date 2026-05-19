# ruvector-cnn/src/simd

Generic FP SIMD kernels used by the FP layers (separate from `kernels/` which is INT8-only).

## Files

- `mod.rs` — arch dispatch entrypoint.
- `scalar.rs` — portable scalar reference.
- `avx2.rs` — x86 AVX2.
- `neon.rs` — ARM NEON.
- `wasm.rs` — WASM SIMD128.
- `quantize.rs` — SIMD-accelerated quantize/dequantize used by `quantize/tensor.rs`.
- `winograd.rs` — Winograd convolution implementation.
