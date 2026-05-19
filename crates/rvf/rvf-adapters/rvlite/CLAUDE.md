# rvf-adapter-rvlite

Lightweight embedded vector-store adapter for RVF Core Profile. Minimal ergonomic API for WASM / edge / embedded use where the full DB is overkill. No metadata, no filters, no namespaces — just vectors with IDs.

## Layout

- `Cargo.toml` — name `rvf-adapter-rvlite`. Deps: `rvf-runtime`, `rvf-types` (`std`). Dev: `tempfile`. Smallest dependency surface of any adapter.
- `src/lib.rs` — public `RvliteCollection`, `RvliteConfig`, `RvliteMetric` re-exports.
- `src/collection.rs` — `RvliteCollection::{create, open, add, search}`.
- `src/config.rs` — `RvliteConfig::new(path, dim).with_metric(...)`, `RvliteMetric` enum (Cosine/L2/Dot).
- `src/error.rs` — `RvliteError`.

## Related

- `../../rvf-runtime`, `../../rvf-types`
- Used by `../../tests/rvf-integration` for round-trip / lifecycle tests.
