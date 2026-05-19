# ruvector-bench/src/bin

One Rust binary per benchmark category. Each is registered as a `[[bin]]` in `../../Cargo.toml`.

## Files

- `ann_benchmark.rs` → `ann-benchmark` — ANN-Benchmarks-compatible harness (recall@k vs QPS curves on standard datasets).
- `agenticdb_benchmark.rs` → `agenticdb-benchmark` — AgenticDB-shaped mixed read/write/learn workload.
- `latency_benchmark.rs` → `latency-benchmark` — p50/p95/p99/p999 search-latency profiling.
- `memory_benchmark.rs` → `memory-benchmark` — peak/working-set memory under load.
- `comparison_benchmark.rs` → `comparison-benchmark` — head-to-head vs external vector DBs.
- `profiling_benchmark.rs` → `profiling-benchmark` — runs paired with perf/flamegraph drivers in `../../../profiling/scripts/`.
