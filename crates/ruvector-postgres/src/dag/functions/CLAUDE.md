# ruvector-postgres/src/dag/functions

SQL function implementations for neural DAG learning. Each file is a topic-grouped set of pgrx `#[pg_extern]` functions.

## Files

- `mod.rs` — Re-exports across submodules.
- `analysis.rs` — Query-trace analysis functions.
- `attention.rs` — Attention-based learning functions.
- `config.rs` — Configuration setters/getters.
- `healing.rs` — Self-healing trigger functions.
- `learning.rs` — Learning loop control.
- `patterns.rs` — Pattern extraction / retrieval.
- `qudag.rs` — QuDAG (quantum DAG) variants.
- `status.rs` — Status reporting.
- `trajectories.rs` — Trajectory tracking SQL surface.
