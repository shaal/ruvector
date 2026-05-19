# ruvector-cnn/src/kernels

Low-level INT8 conv kernels with per-arch backends. Used by the quantized layer set in `src/layers/`.

## Files

- `mod.rs` — kernel selection + arch dispatch.
- `int8_avx2.rs` — x86 AVX2 backend.
- `int8_neon.rs` — ARM NEON backend.
- `int8_wasm.rs` — WASM SIMD128 backend.
- `int8_scalar.rs` — portable scalar fallback.

Cross-arch equivalence is enforced by `tests/kernel_equivalence.rs`.
