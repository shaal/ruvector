# cognitum-gate-kernel/benches

Criterion benchmarks for the coherence-gate tile kernel.

- `benchmarks.rs` — registered as the `[[bench]] name = "benchmarks"` target in `Cargo.toml` (harness = false). Exercises `ingest_delta`, `tick`, and witness-fragment emission against synthetic shards.

See parent `../CLAUDE.md` for crate-level context.
