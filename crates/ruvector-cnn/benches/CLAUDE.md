# ruvector-cnn/benches

Criterion benchmarks.

## Files

- `cnn_benchmarks.rs` — full forward-pass throughput across layer types and SIMD backends.
- `int8_bench.rs` — INT8 kernel throughput vs FP32 baseline.

Run via `cargo bench -p ruvector-cnn`.
