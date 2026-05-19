# mcp-brain-server/src/bin

Auxiliary binaries (declared in crate `Cargo.toml`).

## Files

- `ruvbrain_sse.rs` — `ruvbrain-sse` binary: Server-Sent Events streaming variant of the brain (uses `axum` + `tokio-stream` + `async-stream`).
- `ruvbrain_worker.rs` — `ruvbrain-worker` binary: background worker (Pub/Sub-driven ingest, training, drift jobs).
- `local.rs` — `mcp-brain-server-local` binary (gated by `local` feature, uses bundled SQLite via `rusqlite`).

Main HTTP binary lives at `src/main.rs`.
