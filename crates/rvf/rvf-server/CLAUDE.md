# rvf-server

TCP / HTTP / WebSocket streaming server over `rvf-runtime`. Exposes an HTTP REST API plus a binary TCP streaming protocol for inter-agent vector exchange. Ships an `rvf-server` binary.

## Layout

- `Cargo.toml` — `[[bin]] name = "rvf-server"`. Deps: `rvf-runtime`, `rvf-types`, `tokio` (full), `axum` (with `ws`), `serde`/`serde_json`, `clap` (derive).
- `src/lib.rs` — `ServerConfig { http_port, ... }` + `SharedStore = Arc<Mutex<RvfStore>>` plumbing.
- `src/main.rs` — clap-driven server binary entry.
- `src/http.rs` — axum HTTP REST routes.
- `src/tcp.rs` — binary TCP streaming protocol.
- `src/ws.rs` — WebSocket endpoint over axum.
- `src/error.rs` — server error type.

## Public API

`ServerConfig`, the router/listener entry points; binary is the typical consumer.

## Related

- `../rvf-runtime` — the store served
- `../rvf-cli` `serve` subcommand (optional dep on this crate)
- Sibling: `../../ruvector-server` — alternative axum server over `ruvector-core`
