# rvagent-acp

Agent Communication Protocol (ACP) server for rvAgent. Provides an axum-based
HTTP server with authentication, rate limiting, and body-size enforcement per
ADR-099 and ADR-103 C6.

## Layout

- `Cargo.toml` — lib + bin (`src/main.rs`).
- `src/lib.rs` — module roots; exposes `agent`, `auth`, `server`, `types`.
- `src/main.rs` — standalone server binary entry.
- `src/server.rs` — axum server, route wiring, middleware stack.
- `src/agent.rs` — agent registry / dispatch.
- `src/auth.rs` — authentication (bearer/JWT-style).
- `src/types.rs` — ACP DTOs.
- `tests/` — integration tests for the server.

## Related

- ADR-099 (ACP), ADR-103 C6.
- Typically composed with `rvagent-a2a` in the same axum binary.
