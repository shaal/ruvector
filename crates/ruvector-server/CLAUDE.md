# ruvector-server

High-performance REST API server for ruvector vector databases, built on `axum`. Exposes collection/point CRUD plus health endpoints over HTTP. Companion npm wrapper publishes as `@ruvector/server`.

## Layout

- `Cargo.toml` — deps: `ruvector-core`, `axum` (json/multipart), `tokio` (full), `tower`, `tower-http` (cors/trace/compression), `dashmap`, `parking_lot`, `uuid`, `tracing`. Lints relaxed (research-tier).
- `package.json` — npm metadata for `@ruvector/server` wrapping the binary.
- `src/lib.rs` — `Config { host, port, enable_cors, enable_compression }` with sensible defaults (`127.0.0.1:6333`), plus router assembly (CORS, compression, tracing).
- `src/state.rs` — `AppState { collections: Arc<DashMap<String, Arc<VectorDB>>> }` — shared application state.
- `src/error.rs` — `Error`/`Result` types; axum `IntoResponse` impl.
- `src/routes/` — HTTP route handlers (see `src/routes/CLAUDE.md`).

## Public API

- `Config`, `AppState`, `Error`, `Result`
- Router builder (from `lib.rs`)

## Related

- `../ruvector-core` — backing vector DB
- `../ruvector-metrics` — Prometheus integration (sibling crate, can be wired into the router)
- `../rvf/rvf-server` — alternative RVF-backed HTTP/TCP server
