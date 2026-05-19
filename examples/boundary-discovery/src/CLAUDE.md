# boundary-discovery/src

Source for the synthetic AR(1) boundary-discovery binary.

## Files
- `main.rs` - Generates a 4000-sample series split into a high-autocorrelation regime (phi=0.95) and a low-autocorrelation regime (phi=0.05) at sample 2000, both with unit marginal variance. Builds a 40-window temporal coherence graph and locates the boundary via `estimate_fiedler` + `MinCutBuilder`, with 100 null permutations for significance.

## Constants
- `NUM_SAMPLES=4000`, `WINDOW_SIZE=100`, `TRUE_BOUNDARY=2000`, `NULL_PERMS=100`, `SEED=42`.

## Related
- Parent: `../CLAUDE.md`.
- Other boundary-discovery binaries share the same `ruvector-mincut` + `ruvector-coherence` pattern.
