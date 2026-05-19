# ruvector-sparse-inference/benches

Criterion benchmarks.

- `sparse_inference_bench.rs` — registered as `[[bench]]`. End-to-end sparse
  inference throughput.
- `simd_kernels.rs` — micro-benchmarks for the SIMD kernels in `src/backend/cpu.rs`
  (AVX2 / SSE4.1 / NEON).
