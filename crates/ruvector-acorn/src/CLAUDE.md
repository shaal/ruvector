# ruvector-acorn/src

ACORN source. Modules implement the predicate-agnostic filtered HNSW algorithm.

- `lib.rs` — module roots, public re-exports.
- `index.rs` — `FilteredIndex` trait, `FlatFilteredIndex`, `AcornIndex1`, `AcornIndexGamma`,
  and `recall_at_k` helper.
- `graph.rs` — `AcornGraph` HNSW graph with denser per-node edges (gamma * M).
- `search.rs` — beam-search that expands all neighbours regardless of predicate.
- `dist.rs` — L2-squared distance kernels (auto-vectorised).
- `error.rs` — `AcornError` enum.
- `main.rs` — `acorn-demo` binary (synthetic data, prints recall/QPS).
