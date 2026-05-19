# ruvector-rabitq/benches

Criterion micro-benchmarks for RaBitQ distance kernels and index search.
Run with `cargo bench -p ruvector-rabitq`.

## Files
- `rabitq_bench.rs` - the `[[bench]]` declared in Cargo.toml
  (`harness = false`). Measures symmetric vs asymmetric estimator
  throughput, rotation cost, and end-to-end search latency. See
  `../BENCHMARK.md` for representative numbers.
