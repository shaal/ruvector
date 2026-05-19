# ruvector-sparse-inference/src/backend

Per-target compute backends behind a common backend trait.

- `mod.rs` — backend trait and dispatch.
- `cpu.rs` — CPU backend with SIMD (AVX2 / SSE4.1 / NEON) kernels.
- `npu.rs` — NPU backend stub / accelerator integration.
- `wasm.rs` — WebAssembly SIMD backend.

Tested by `tests/backend_simd_tests.rs` and `benches/simd_kernels.rs`.
