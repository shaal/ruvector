# ruvector-hyperbolic-hnsw-wasm/src

Single-file WASM binding crate.

## Files

- `lib.rs` — wasm-bindgen bindings only. Wraps `ruvector_hyperbolic_hnsw::{HyperbolicHnsw, HyperbolicHnswConfig, PoincareConfig, ShardedHyperbolicHnsw, TangentCache, poincare_distance, mobius_add, mobius_scalar_mult, exp_map, log_map, frechet_mean, project_to_ball, DEFAULT_CURVATURE, EPS}`. Sets up `console_error_panic_hook` when the default feature is on.
