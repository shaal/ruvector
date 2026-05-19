# 07-causal-emergence

Standalone research crate: Hierarchical Causal Consciousness (HCC)
framework with O(log n) emergence detection. Pure Rust, no runtime
deps.

## Files

- `Cargo.toml` — standalone `[workspace]`; package `causal-emergence`.
  Dev-dep `criterion`. Bench `causal_emergence_bench`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `SUMMARY.md`,
  `mathematical_framework.md` — theory.
- `rust-toolchain.toml` — pinned toolchain for reproducibility.
- `Cargo.lock` — pinned.
- `src/lib.rs` — re-exports.
- `src/causal_hierarchy.rs` — hierarchical causal structure.
- `src/coarse_graining.rs` — macroscale coarse-graining.
- `src/effective_information.rs` — Tononi-Hoel effective information.
- `src/emergence_detection.rs` — O(log n) emergence detector.
- `benches/causal_emergence_bench.rs` — Criterion suite.

## Build / Bench

```bash
cd examples/exo-ai-2025/research/07-causal-emergence
cargo build --release
cargo bench
```

## Related

- `../../crates/exo-exotic/src/experiments/causal_emergence.rs`
- `../../../ecosystem-consciousness/` — uses similar causal-emergence
  scoring on food webs
