# 05-memory-mapped-neural-fields

Standalone research crate: memory-mapped (mmap-backed) neural fields
plus tiered RAM/SSD storage, lazy activation, and prefetch prediction —
aimed at petabyte-scale cognition on commodity hardware.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `demand-paged-cognition`. Runtime dep: `memmap2`. Optional features:
  `tokio` (async), `serde`/`bincode` (checkpointing), `metrics`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `EXECUTIVE_SUMMARY.md`,
  `architecture.md` — design + theory.
- `Cargo.lock` — pinned.
- `src/lib.rs` — re-exports.
- `src/mmap_neural_field.rs` — mmap-backed neural field.
- `src/lazy_activation.rs` — lazy activation evaluation.
- `src/tiered_memory.rs` — RAM / SSD tiered storage.
- `src/prefetch_prediction.rs` — predictive prefetch.
- `benches/neural_field_bench.rs`, `benches/prefetch_bench.rs` —
  Criterion suites.
- `examples/basic_usage.rs`, `examples/petabyte_scale.rs` — runnable
  demos.

## Build / Run

```bash
cd examples/exo-ai-2025/research/05-memory-mapped-neural-fields
cargo run --release --example basic_usage
cargo run --release --example petabyte_scale
cargo bench
```

## Related

- `../../crates/exo-exotic/src/experiments/memory_mapped_fields.rs`
