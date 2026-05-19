# scipix/src/api

HTTP API surface for scipix, built on axum + tower.

## Files

- `mod.rs` - Module surface.
- `routes.rs` - Route table.
- `handlers.rs` - Endpoint handlers (OCR, status, jobs).
- `state.rs` - Shared `AppState`.
- `requests.rs`, `responses.rs` - Validated request/response DTOs.
- `middleware.rs` - Auth, logging, rate-limiting, CORS, compression.
- `jobs.rs` - Async job runner for long OCR tasks.

## Related

- Bin: `../bin/server.rs`.
- Docs: `../../docs/13_API_SERVER.md`, `../../docs/API_SERVER.md`.
