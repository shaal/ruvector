# ruvector-attention/benches

Criterion benchmarks for attention kernels.

## Files

- `attention_bench.rs` — focused micro-benchmarks for the core kernels (scaled-dot-product, multi-head).
- `attention_benchmarks.rs` — broader suite covering MLA, flash, sparse, hyperbolic variants. Also used as the `bench_runner` binary configured in `Cargo.toml`.

Run via `cargo bench -p ruvector-attention`.
