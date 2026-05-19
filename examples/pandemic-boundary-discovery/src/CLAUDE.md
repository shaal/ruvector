# pandemic-boundary-discovery/src

## Files

- `main.rs` — single-file binary. Synthesizes 300 days of 8 public
  health signals across four phases (baseline / silent spread /
  exponential growth / declared outbreak), bins into 10-day windows
  (`N_WIN = 30`), computes the 28-pair upper-triangle correlation
  matrix per window, runs min-cut + Fiedler segmentation, and shows
  detection precedes the declared outbreak day (`DECLARED = 215`).

## Related

- `../Cargo.toml` — dependency wiring
- `../../../crates/ruvector-mincut/`,
  `../../../crates/ruvector-coherence/`
