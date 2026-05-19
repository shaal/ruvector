# rvf-adapter-rvlite/src

Source.

## Files

- `lib.rs` — public re-exports + design-philosophy docs (simple/small/safe).
- `collection.rs` — `RvliteCollection`: `create`, `open`, `add(id, vector)`, `search`.
- `config.rs` — `RvliteConfig { path, dim }` with `with_metric(RvliteMetric)`.
- `error.rs` — `RvliteError` via `thiserror`.
