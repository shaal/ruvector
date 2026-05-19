# seizure-therapeutic-sim

Closed-loop seizure detection + therapeutic-response simulation: runs
TWO 16-channel EEG simulations side-by-side (CONTROL vs. INTERVENTION),
detects a pre-ictal -> seizure transition in the control arm, and in
the intervention arm applies alpha-band entrainment at the detected
moment. The entrainment delays the seizure by ~60 s and in ~30 % of
parameter regimes reverses the drift entirely. Working CLI demo.

## Important files

- `Cargo.toml` — `publish = false`; deps:
  - `ruvector-mincut` (`../../crates/ruvector-mincut`) with `exact`
  - `ruvector-coherence` (`../../crates/ruvector-coherence`) with
    `spectral`
  - `rand 0.8`
- `src/main.rs` — single-file binary; constants `NCH = 16`, `DUR = 600`,
  `SR = 256`, `WIN_S = 10`, control-phase boundaries `P1`/`P2`/...,
  seizure-detection threshold `AMP_THR = 5.0`, permutation null
  `NULL_N = 80`. Seed `42_0911`.

## Run

```bash
cargo run -p seizure-therapeutic-sim --release
```

## Tech stack

- Pure Rust; deterministic with seeded `StdRng`.
- `ruvector_coherence::spectral` (Fiedler) +
  `ruvector_mincut::MinCutBuilder`.

## Related siblings

- `../health-boundary-discovery/`,
  `../pandemic-boundary-discovery/` — the same boundary-detection
  pipeline applied to other domains
- `../subpolynomial-time/` — underlying min-cut algorithm demo
