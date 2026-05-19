# ruvector-postgres/src/hyperbolic

Hyperbolic embeddings — Poincaré ball and Lorentz hyperboloid models for hierarchical embeddings. Default curvature `-1.0`, epsilon `1e-8`.

## Files

- `mod.rs` — Re-exports `PoincareBall`, `LorentzModel`; defines `DEFAULT_CURVATURE` and `EPSILON`.
- `poincare.rs` — Poincaré ball model arithmetic.
- `lorentz.rs` — Lorentz hyperboloid model arithmetic.
- `operators.rs` — pgrx SQL operator wrappers.

## Pointers

- Pure-Rust standalone crate: `ruvector-hyperbolic-hnsw`.
