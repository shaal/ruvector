# subpolynomial-time

Comprehensive demo of the `ruvector-mincut` crate's subpolynomial-time
dynamic minimum-cut algorithm. Covers seven scenarios: basic min-cut,
dynamic insert/delete, exact vs. approximate modes, real-time
monitoring, network resilience, performance scaling, and "Vector-Graph
Fusion" with brittleness detection. Working CLI demo.

## Important files

- `Cargo.toml` — package `subpolynomial-time-mincut-demo`; dep
  `ruvector-mincut` (`../../crates/ruvector-mincut`) with the
  `monitoring` feature; plus workspace-wide lint relaxations.
- `src/main.rs` — orchestrator that walks each scenario and prints
  banners. Imports the local `fusion` module for the brittleness demo.

## Run

```bash
cargo run -p subpolynomial-time-mincut-demo --release
```

## Tech stack

- Pure Rust; deterministic via `rand::prelude::*`.
- `ruvector_mincut::prelude` + `EventType`, `MonitorBuilder`.

## Related siblings

- `../health-boundary-discovery/`,
  `../pandemic-boundary-discovery/`,
  `../seizure-therapeutic-sim/` — applied demos of the same min-cut /
  spectral pipeline
