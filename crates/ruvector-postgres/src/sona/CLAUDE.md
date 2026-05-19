# ruvector-postgres/src/sona

Sona self-learning module — Micro-LoRA trajectories and EWC++ for PostgreSQL. Wraps `ruvector-sona` and caches per-table+dimension engines.

## Files

- `mod.rs` — `engine_key(table_name, dim)` helper; global `SONA_ENGINES: DashMap<String, Arc<SonaEngine>>`.
- `operators.rs` — pgrx SQL function surface (engine create, begin/end trajectory, apply LoRA, persist/load, etc.).

## Pointers

- Backbone: `ruvector-sona` (sibling crate at `crates/sona`).
- DAG-level integration: `../dag/`.
