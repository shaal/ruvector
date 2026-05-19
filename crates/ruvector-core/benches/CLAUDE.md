# ruvector-core/benches

Criterion benchmarks for the core engine.

## Files

- `batch_operations.rs` — bulk insert / update / delete throughput.
- `bench_memory.rs` — allocator / mmap behavior.
- `bench_simd.rs` — distance kernel microbenchmarks.
- `comprehensive_bench.rs` — end-to-end workload (insert + search).
- `distance_metrics.rs` — cosine / L2 / IP comparison.
- `hnsw_search.rs` — HNSW query QPS (~2.5K qps on 10K vectors per lib.rs).
- `quantization_bench.rs` — scalar / int4 / PQ / binary compression and distance.
- `real_benchmark.rs` — realistic dataset benchmark.
