# ruvector-hyperbolic-hnsw/src

Source for hyperbolic-HNSW vector search.

## Files

- `lib.rs` — Crate doc, module declarations, re-exports.
- `poincare.rs` — Poincaré ball model — Möbius addition, exp/log maps, distance.
- `tangent.rs` — Tangent-space pruning (cheap Euclidean prefilter).
- `hnsw.rs` — `HyperbolicHnsw` index, `HyperbolicHnswConfig`, insert/search.
- `shard.rs` — Per-shard curvature control.
- `error.rs` — Error type.
