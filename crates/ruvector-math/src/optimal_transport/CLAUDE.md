# ruvector-math/src/optimal_transport

Optimal transport distances and solvers between probability distributions.

- `mod.rs` — re-exports.
- `config.rs` — solver configuration types.
- `sliced_wasserstein.rs` — O(n log n) Sliced Wasserstein via random 1D projections (`SlicedWasserstein`).
- `sinkhorn.rs` — log-stabilised entropic Sinkhorn algorithm (`SinkhornSolver`).
- `gromov_wasserstein.rs` — Gromov-Wasserstein for cross-space structure comparison.

Use cases: cross-lingual retrieval, image-region matching, time-series pattern matching, document similarity via word-embedding distributions.

See `../CLAUDE.md`.
