# rvf/benches

Cargo crate (`rvf-benches`, `publish = false`) that runs Criterion benchmarks against the entire RVF crate family.

## Layout

- `Cargo.toml` — `[[bench]] name = "rvf_benchmarks" harness = false`. Deps: every core RVF crate (`rvf-types`, `rvf-wire`, `rvf-manifest`, `rvf-index`, `rvf-quant`, `rvf-crypto`, `rvf-runtime`) + `criterion`, `rand`, `tempfile`, `ed25519-dalek`.
- `benches/rvf_benchmarks.rs` — wire-format, indexing, distance, quantization, manifest, runtime, crypto throughput/latency vs. acceptance targets in `docs/research/rvf/benchmarks/`. Uses an in-file LCG for deterministic data.
