# rvm-coherence

Real-time coherence scoring and Phi computation for the RVM microhypervisor (ADR-139). Coherence is the first-class resource-allocation signal: partitions with higher coherence receive more CPU time and memory grants.

Pipeline: sensor data → Phi → score update → scheduler feedback.

`#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`.

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `rvm-partition`; optional `rvm-sched` (feature `sched`).
- `src/lib.rs` — module wiring.
- `src/graph.rs` — fixed-size adjacency for the inter-partition communication topology.
- `src/scoring.rs` — coherence score (internal / total weight ratio).
- `src/pressure.rs` — cut-pressure and split / merge signals.
- `src/mincut.rs` — budgeted approximate minimum cut (Stoer-Wagner heuristic).
- `src/adaptive.rs` — adaptive recomputation frequency based on CPU load.
- `src/bridge.rs`, `src/engine.rs` — engine glue and scheduler bridge.

See `../CLAUDE.md`.
