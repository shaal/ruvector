# ruvector-sparsifier/src

Source modules for dynamic spectral graph sparsification.

## Files

- `lib.rs` — crate root, quickstart docs, re-exports.
- `types.rs` — `SparsifierConfig` and value/return types shared across modules.
- `traits.rs` — `Sparsifier` trait (build / insert_edge / delete_edge / audit / sparsifier accessor).
- `graph.rs` — `SparseGraph` host graph type (`from_edges`, accessors).
- `sparsifier.rs` — `AdaptiveGeoSpar` main implementation orchestrating backbone + sampling + audit.
- `backbone.rs` — spanning-forest construction guaranteeing global connectivity (Step 1).
- `importance.rs` — random-walk-based effective-resistance estimation (Step 2).
- `sampler.rs` — spectral edge sampling proportional to `w * R_eff * log(n) / eps^2` (Step 3).
- `audit.rs` — periodic random quadratic-form drift probes (Step 4); `audit()` returns max-error report.
- `error.rs` — crate error enum.
