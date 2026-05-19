Output directory for ad-hoc Rust/Python benchmark runs comparing ruvector's HNSW search against baselines. Files here are regenerated artifacts, not source.

Contents:
- `comparison_benchmark.{csv,json,md}` - cross-implementation comparison (ruvector vs Python baseline vs brute force) at 384D / 10k vectors.
- `latency_benchmark.{csv,json,md}` - p50/p95/p99/p99.9 latency results for the same dataset.

The `.md` and `.csv` files are human-readable summaries; the `.json` is the raw machine-readable form consumed by downstream reporting. Producers live under `../benchmarks/` (the larger TS/k6 suite) and `../crates/ruvector-bench/`. Newer quantization-focused results live in `../benchmarks/vector-search/results/`.
