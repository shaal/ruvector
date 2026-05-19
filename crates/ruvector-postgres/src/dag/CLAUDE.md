# ruvector-postgres/src/dag

Neural DAG learning for PostgreSQL query optimization. Integrates the SONA engine with the PostgreSQL query planner to provide learned query optimization.

## Files

- `mod.rs` — Declares `functions` and `state` submodules; re-exports `DagConfig`, `DagState`, `DAG_STATE`.
- `state.rs` — Global `DagState` + `DagConfig`, lazy `DAG_STATE`.
- `extension.rs` — Extension registration entrypoints.
- `guc.rs` — GUC (Grand Unified Config) parameter definitions.
- `hooks.rs` — PostgreSQL planner hooks.
- `worker.rs` — Background worker for DAG learning.
- `functions/` — SQL function implementations (analysis, attention, config, healing, learning, patterns, qudag, status, trajectories).

## Pointers

- Wraps `ruvector-sona` (see `../sona/`).
- Hooks tie into `../healing/` and `../learning/`.
