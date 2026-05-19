# health-boundary-discovery/src

## Files

- `main.rs` — single-file binary that synthesizes 90 days of half-day
  wearable observations across four health phases (healthy /
  overtraining / sick / recovery), computes per-window correlation
  matrices over 8 features, builds graphs via `MinCutBuilder`,
  estimates the Fiedler vector via `estimate_fiedler`, and prints
  detected boundaries vs. ground-truth `TRUE_B` along with a
  permutation null distribution (`NULL_PERMS = 100`).

## Related

- `../Cargo.toml` — dependency wiring
- `../../../crates/ruvector-mincut/`,
  `../../../crates/ruvector-coherence/`
