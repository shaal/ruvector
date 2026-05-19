# 04-sparse-persistent-homology

Standalone research crate: sub-cubic persistent homology with SIMD
acceleration for real-time consciousness measurement. Pure Rust (no
runtime deps); leans on apparent-pairs optimization and a streaming
filtration.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `sparse-persistent-homology`. No runtime deps; dev-deps `criterion`,
  `rand`. Bench `sparse_homology_bench`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `complexity_analysis.md`
  — theory + complexity discussion.
- `Cargo.lock` — pinned.
- `src/lib.rs` — public surface.
- `src/apparent_pairs.rs` — apparent-pairs optimization.
- `src/sparse_boundary.rs` — sparse boundary matrices.
- `src/simd_filtration.rs`, `src/simd_matrix_ops.rs` — SIMD inner
  loops.
- `src/streaming_homology.rs` — streaming filtration / homology updates.
- `benches/sparse_homology_bench.rs` — Criterion suite.

## Build / Bench

```bash
cd examples/exo-ai-2025/research/04-sparse-persistent-homology
cargo build --release
cargo bench
```

## Related

- `../../crates/exo-hypergraph/src/sparse_tda.rs` — production
  counterpart
- `../../crates/exo-exotic/src/experiments/sparse_homology.rs`
