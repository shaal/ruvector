# ruvector-postgres/benches

Criterion benchmarks for the PostgreSQL extension.

## Files

- `distance_bench.rs` — SIMD distance kernels (AVX-512/AVX2/NEON/scalar).
- `e2e_bench.rs` — End-to-end query latency.
- `hybrid_bench.rs` — Hybrid BM25 + vector search.
- `index_bench.rs` — HNSW + IVFFlat index build/query.
- `integrity_bench.rs` — Mincut/integrity gating performance.
- `quantization_bench.rs` — Scalar/Product/Binary quantization speed.
- `quantized_distance_bench.rs` — Distance kernels on quantized vectors.

## Subdirectories

- `scripts/run_benchmarks.sh` — Helper invocation.
- `sql/benchmark_workload.sql`, `sql/quick_benchmark.sql` — SQL-level workloads driven against a running extension.
