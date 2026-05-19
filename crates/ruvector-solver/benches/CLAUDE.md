# ruvector-solver/benches

Criterion benchmarks.

- `solver_baseline.rs` — baseline (dense / unoptimized) reference.
- `solver_neumann.rs` — Neumann-series solver throughput.
- `solver_push.rs` — forward/backward push solvers.
- `solver_cg.rs` — conjugate gradient.
- `solver_e2e.rs` — end-to-end solve with router-driven algorithm selection.
