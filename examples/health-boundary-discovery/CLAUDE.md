# health-boundary-discovery

Research demo applying graph-structural boundary detection to wearable
sensor data (HR, HRV, steps, sleep). Shows that the *correlation
structure* between metrics shifts BEFORE any single metric crosses a
clinical threshold — so a min-cut + spectral-Fiedler pipeline can spot
overtraining onset days before a doctor would. Working CLI demo; no
external data needed (synthetic 90-day series with known boundaries).

## Important files

- `Cargo.toml` — `publish = false`; deps:
  - `ruvector-mincut` (`../../crates/ruvector-mincut`) with `exact`
  - `ruvector-coherence` (`../../crates/ruvector-coherence`) with
    `spectral`
  - `rand 0.8`
  Plus extensive workspace-wide lint relaxations.
- `src/main.rs` — single-file binary; constants drive the simulation
  (`N_OBS = 180`, `WINDOW = 6`, four phases ending at half-day indices
  60/100/130, `NULL_PERMS = 100` permutation null).

## Run

```bash
cargo run -p health-boundary-discovery --release
```

## Tech stack

- Pure Rust, no async; deterministic via `StdRng` seed `118`.
- `ruvector_coherence::spectral::{estimate_fiedler, CsrMatrixView}`,
  `ruvector_mincut::MinCutBuilder`.

## Related siblings

- `../pandemic-boundary-discovery/` — same algorithm applied to public
  health monitoring streams
- `../seizure-therapeutic-sim/` — same algorithm on EEG channels with
  a therapeutic intervention arm
- `../subpolynomial-time/` — deeper dive into the underlying min-cut
  algorithm
