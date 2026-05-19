# rvagent-acp/src

Source for the ACP server.

- `lib.rs` — module roots (`agent`, `auth`, `server`, `types`).
- `main.rs` — binary entry that boots the server.
- `server.rs` — axum router, rate-limit + body-size middleware, ACP handlers.
- `agent.rs` — agent registry / dispatch backing the HTTP endpoints.
- `auth.rs` — bearer-token / JWT authentication.
- `types.rs` — request/response DTOs.
