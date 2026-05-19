# prime-radiant/src/hyperbolic

Hierarchy-aware Poincare hyperbolic energy: maps substrate depth into a Poincare ball to give exponentially better separation of taxonomic hierarchies.

## Files

- `mod.rs` — module entry.
- `config.rs` — curvature, embedding dim, depth schedule.
- `depth.rs` — depth assignment / propagation along the substrate.
- `energy.rs` — Poincare-based per-edge energy variant.
- `adapter.rs` — bridges into `ruvector-hyperbolic-hnsw`.

## Related

- `crates/ruvector-hyperbolic-hnsw` — underlying Poincare HNSW.
