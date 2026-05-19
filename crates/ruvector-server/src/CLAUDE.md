# ruvector-server/src

Source layout.

## Files

- `lib.rs` — server `Config`, default values, router assembly with `CorsLayer`, `CompressionLayer`, `TraceLayer`. Re-exports `Error`, `Result`, `AppState`.
- `state.rs` — `AppState` holding `Arc<DashMap<String, Arc<VectorDB>>>` keyed by collection name.
- `error.rs` — server error type and axum `IntoResponse` impl.
- `routes/` — endpoint handlers; see `routes/CLAUDE.md`.
