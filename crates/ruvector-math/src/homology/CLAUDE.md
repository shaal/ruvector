# ruvector-math/src/homology

Persistent homology and topological data analysis. Tracks topological features (components, loops, voids) across a filtration to detect long-term drift and monitor coherence of attention patterns.

- `mod.rs` — re-exports.
- `simplex.rs` — simplex / simplicial-complex types.
- `filtration.rs` — filtration construction (Vietoris-Rips etc.).
- `persistence.rs` — persistence-diagram computation, Betti numbers.
- `distance.rs` — bottleneck / Wasserstein distances between persistence diagrams.

Integrates with the mincut spine: complements partition-based metrics with shape-based drift signals. See `../CLAUDE.md`.
