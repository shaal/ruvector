# 09-hyperbolic-attention

Standalone research crate: hyperbolic attention networks providing
O(log n) hierarchical reasoning capacity via Poincaré / Lorentz
embeddings. No runtime deps.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `hyperbolic-attention`. Dev-deps `approx`, `criterion`. Bench
  `hyperbolic_ops`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `RESEARCH_SUMMARY.md`,
  `geometric_foundations.md` — theory.
- `Cargo.lock` — pinned.
- `src/lib.rs` — re-exports.
- `src/poincare_embedding.rs` — Poincaré ball embeddings.
- `src/lorentz_model.rs` — Lorentz / hyperboloid model.
- `src/curvature_adaptation.rs` — adaptive curvature.
- `src/hyperbolic_attention.rs` — attention in hyperbolic space.
- `benches/hyperbolic_ops.rs` — Criterion suite.
- `tests/debug_tests.rs` — diagnostic / debug tests.

## Build / Test / Bench

```bash
cd examples/exo-ai-2025/research/09-hyperbolic-attention
cargo build --release
cargo test
cargo bench
```

## Related

- `../../crates/exo-manifold/` — production manifold engine
