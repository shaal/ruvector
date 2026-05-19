# ruvector-profiler

Memory, power, and latency profiling hooks with CSV emitters for benchmarking
attention mechanisms (and other ruvector workloads). Lightweight - no async,
no allocators, just measurement + serialization.

## Important files
- `Cargo.toml` - workspace inheritance. Deps: `serde`, `serde_json`. Dev:
  `tempfile`.
- `src/lib.rs` - module declarations + flat re-export of the public API.
- `src/config_hash.rs` - `BenchConfig` + `config_hash(...)` for stable
  config fingerprints (so CSV rows can be joined across runs).
- `src/csv_emitter.rs` - `ResultRow`, `write_results_csv`, `write_latency_csv`,
  `write_memory_csv`.
- `src/latency.rs` - `LatencyRecord`, `LatencyStats`, `compute_latency_stats`
  (p50/p95/p99/etc).
- `src/memory.rs` - `MemoryTracker`, `MemorySnapshot`, `MemoryReport`,
  `capture_memory`.
- `src/power.rs` - `PowerTracker`, `PowerSource` trait, `MockPowerSource`,
  `PowerSample`, `EnergyResult`.

## Public API surface
Drop-in measurement helpers used by `benches/*.rs` across the workspace:
construct trackers, sample around the code under test, then write CSV for
later analysis.

## Tests / benches
None directly here; consumed by other crates' benches.

## Related
- `ruvector-attention`, `ruvllm/benches/*` use the CSV format produced
  here.
