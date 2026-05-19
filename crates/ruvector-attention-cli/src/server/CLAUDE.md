# ruvector-attention-cli/src/server

axum-based HTTP server for the `Serve` subcommand. Wraps the
`ruvector-attention` engine behind REST handlers with CORS and tracing
middleware (`tower-http`).

## Files
- `mod.rs` - router construction, app state, server bootstrap.
- `handlers.rs` - per-endpoint axum handlers (compute, health, info, etc.).
