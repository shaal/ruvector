# rvf-server/src

Source.

## Files

- `lib.rs` — `ServerConfig` (HTTP port, TCP port, etc.) + `SharedStore = Arc<Mutex<RvfStore>>` glue.
- `main.rs` — `rvf-server` binary entry; clap parses flags, sets up tokio runtime, binds HTTP/TCP/WS listeners.
- `http.rs` — axum REST handlers.
- `tcp.rs` — binary TCP streaming protocol implementation.
- `ws.rs` — WebSocket endpoint built on axum.
- `error.rs` — server error type with axum `IntoResponse` impl.
