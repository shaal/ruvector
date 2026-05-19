# ruvector-attention/src/transport

Optimal-transport / Wasserstein helpers used by curvature-aware attention.

## Files

- `mod.rs` — module entry.
- `centroid_ot.rs` — centroid-based optimal transport.
- `sliced_wasserstein.rs` — sliced Wasserstein distance approximations.
- `cached_projections.rs` — cached random projections to amortise sliced-OT cost.
