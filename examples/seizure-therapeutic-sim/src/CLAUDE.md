# seizure-therapeutic-sim/src

## Files

- `main.rs` — single-file binary that runs two parallel 16-channel,
  256 Hz, 600 s EEG simulations. For each 10 s window it computes the
  cross-channel correlation pattern (120 pairs + per-channel features,
  `NFEAT = NPAIRS + NCH * 4`), runs the Fiedler / min-cut detector,
  and at the detected seizure boundary the INTERVENTION arm injects
  alpha-band entrainment. Compares outcomes between arms and reports
  seizure delay / suppression.

## Related

- `../Cargo.toml` — dependency wiring
- `../../../crates/ruvector-mincut/`,
  `../../../crates/ruvector-coherence/`
