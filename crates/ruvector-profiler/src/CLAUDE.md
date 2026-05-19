# ruvector-profiler/src

Profiling primitives + CSV serialization. Each module owns one measurement
domain.

## Files
- `lib.rs` - flat re-exports.
- `config_hash.rs` - `BenchConfig` + `config_hash(...)` for stable
  benchmark-config fingerprints used as join keys in CSVs.
- `csv_emitter.rs` - `ResultRow` and writers: `write_results_csv`,
  `write_latency_csv`, `write_memory_csv`.
- `latency.rs` - `LatencyRecord` (samples), `LatencyStats` (aggregates),
  `compute_latency_stats(...)`.
- `memory.rs` - `MemoryTracker`, `MemorySnapshot`, `MemoryReport`,
  `capture_memory(...)`.
- `power.rs` - `PowerSource` trait (impl per backend), `MockPowerSource`,
  `PowerTracker`, `PowerSample`, `EnergyResult`.
