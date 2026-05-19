# ruvector-mincut/src/algorithm

Top-level algorithm implementations selected via `MinCutBuilder`.

## Files

- `mod.rs` — module wiring + dispatcher.
- `approximate.rs` — (1+ε)-approximate dynamic min-cut via graph sparsification.
- `replacement.rs` — exact replacement-based algorithm (used by `MinCutBuilder::exact()`).

See also `subpolynomial/` for the headline subpolynomial-time scheme.
