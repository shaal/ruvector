# ruQu/benches

Criterion benchmarks for `ruqu` syndrome-processing performance.

## Files

- `latency_bench.rs` — End-to-end latency from syndrome ingest to gate decision.
- `memory_bench.rs` — Memory footprint of detector bitmaps and tile fabric.
- `mincut_bench.rs` — Dynamic min-cut algorithm benchmarks.
- `scaling_bench.rs` — Scaling behavior across detector counts and tile counts.
- `syndrome_bench.rs` — Syndrome buffer throughput.
- `throughput_bench.rs` — Aggregate pipeline throughput.

Run via `cargo bench -p ruqu`.
