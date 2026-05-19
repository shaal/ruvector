# prime-radiant/src/mincut

Cognitive partitioning via `ruvector-mincut`: subpolynomial-time n^o(1) mincut for isolating coherence regions.

## Files

- `mod.rs` — module entry.
- `config.rs` — partition strategy + thresholds.
- `adapter.rs` — bridges into `SubpolynomialMinCut` / `CognitiveMinCutEngine` / `WitnessTree`.
- `isolation.rs` — region isolation policy after a cut.
- `metrics.rs` — cut-quality metrics.

## Related

- `crates/ruvector-mincut` — underlying mincut algorithms.
