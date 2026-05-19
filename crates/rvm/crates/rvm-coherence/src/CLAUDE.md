# rvm-coherence/src

- `lib.rs` — crate root.
- `graph.rs` — fixed-size adjacency structure for the partition communication topology.
- `scoring.rs` — coherence score (internal / total edge weight).
- `pressure.rs` — cut pressure and split / merge signal computation.
- `mincut.rs` — budgeted approximate Stoer-Wagner min-cut heuristic.
- `adaptive.rs` — adaptive recompute frequency based on CPU load.
- `engine.rs` — top-level coherence engine.
- `bridge.rs` — bridge to `rvm-sched` for coherence-weighted scheduling feedback (feature `sched`).

See `../CLAUDE.md`.
