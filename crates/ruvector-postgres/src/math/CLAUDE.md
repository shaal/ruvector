# ruvector-postgres/src/math

Math distances and spectral methods module — exposes `ruvector-math` as SQL functions.

## Files

- `mod.rs` — Module declaration (`pub mod operators;`).
- `operators.rs` — pgrx `#[pg_extern]` SQL functions wrapping distances and spectral methods from `ruvector-math`.
