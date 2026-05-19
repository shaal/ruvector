# pandemic-boundary-discovery

Research demo applying graph-structural boundary detection to 8 public
health monitoring signals (e.g., wastewater, ER visits, OTC sales) over
300 simulated days. Finds the outbreak's "silent spread" boundary
~60 days before case counts cross any alarm threshold — the cross-signal
correlation pattern shifts first. Working CLI demo, synthetic data.

## Important files

- `Cargo.toml` — `publish = false`; deps mirror the sibling boundary
  demos:
  - `ruvector-mincut` (`../../crates/ruvector-mincut`) with `exact`
  - `ruvector-coherence` (`../../crates/ruvector-coherence`) with
    `spectral`
  - `rand 0.8`
  Plus the standard workspace lint relaxations.
- `src/main.rs` — single-file binary; constants `DAYS = 300`, `SIG = 8`,
  `WIN = 10`, phase boundaries `P1_END = 150`, `P2_END = 200`,
  `P3_END = 250`, `DECLARED = 215`, `NULL_PERMS = 100`.

## Run

```bash
cargo run -p pandemic-boundary-discovery --release
```

## Tech stack

- Pure Rust, `StdRng` seeded with `42` for determinism.
- `ruvector_coherence::spectral` + `ruvector_mincut::MinCutBuilder`.

## Related siblings

- `../health-boundary-discovery/` — same algorithm on individual
  wearable data
- `../seizure-therapeutic-sim/` — same algorithm on EEG with a
  therapeutic intervention
- `../subpolynomial-time/` — algorithm-level demo of the underlying
  min-cut machinery
