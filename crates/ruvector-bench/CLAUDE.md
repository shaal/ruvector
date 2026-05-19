# ruvector-bench

Comprehensive benchmarking suite for the ruvector vector database family. `publish = false`. Ships six executable bench harnesses plus a shared library of measurement utilities (latency percentiles, recall@k, memory).

## Layout

- `Cargo.toml` — six `[[bin]]` targets (`ann-benchmark`, `agenticdb-benchmark`, `latency-benchmark`, `memory-benchmark`, `comparison-benchmark`, `profiling-benchmark`). Depends on `ruvector-core`, `ruvector-mincut` (canonical), `ruvector-coherence` (spectral).
- `src/lib.rs` — shared `BenchmarkResult` struct (qps, p50/p95/p99/p999, recall@1/10/100, memory MB, build time, metadata). Helpers for dataset generation, percentile calculation.
- `src/bin/` — one binary per benchmark category. See `src/bin/CLAUDE.md`.
- `tests/wasm_stack_bench.rs` — integration tests benchmarking the WASM cognitive stack (canonical min-cut, SCS, witness fragment) against target latencies.
- `scripts/` — `download_datasets.sh`, `run_all_benchmarks.sh`.
- `docs/BENCHMARKS.md` — full user-facing documentation.

## Public API

- `BenchmarkResult` — serialisable benchmark output
- Dataset and workload generators (uniform / normal distributions)

## Related

- `../ruvector-core`, `../ruvector-mincut`, `../ruvector-coherence` — units under test
- `../profiling/` — perf/flamegraph wrappers that consume these binaries
