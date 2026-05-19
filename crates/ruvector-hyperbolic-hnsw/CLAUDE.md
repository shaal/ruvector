# ruvector-hyperbolic-hnsw

Hyperbolic (Poincaré ball) embeddings integrated with HNSW for hierarchy-aware vector search — efficient similarity in non-Euclidean spaces for taxonomies, ontologies, ICD trees, product facets, org charts, and long-tail tags. Tangent-space pruning + per-shard curvature + dual-space (Euclidean fallback) index.

## Important files

- `Cargo.toml` — Standalone-style versioned crate (`version = "0.1.0"`). Features `default = ["simd","parallel"]`, plus `wasm`. Deps: `nalgebra` (exact 0.34.1), `ndarray` (0.17.1), `rayon`, `serde`, `thiserror`, `rand`, `rand_distr`. Dev: `criterion`.
- `Cargo.lock` — Lockfile.
- `src/lib.rs` — Crate root. Quick-start example via `HyperbolicHnsw::default_config()`, `.insert`, `.search`.

## Source modules (`src/`)

- `lib.rs` — Crate doc + re-exports.
- `poincare.rs` — Poincaré ball model: Möbius addition, exp/log maps.
- `tangent.rs` — Tangent-space pruning (cheap Euclidean prefilter before exact hyperbolic ranking).
- `hnsw.rs` — `HyperbolicHnsw` index + `HyperbolicHnswConfig`.
- `shard.rs` — Per-shard curvature support.
- `error.rs` — Crate error type.

## Tests / Benches

- `tests/math_tests.rs` — Validation of Poincaré arithmetic and tangent-space math.
- `benches/hyperbolic_bench.rs` — Index throughput + query latency vs Euclidean baseline.

## Public API

- `HyperbolicHnsw`, `HyperbolicHnswConfig`.
- Math primitives from `poincare` and `tangent` modules.

## Related

- Used by `ruvector-postgres/src/hyperbolic/` (Postgres operators) and `ruvector-graph/src/hybrid/`.
