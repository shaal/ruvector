# sevensense-api

REST, GraphQL, and WebSocket API server for 7sense - the composition root that wires every other `sevensense-*` crate together.

## Files
- `Cargo.toml` - Depends on every other `sevensense-*` crate, plus `axum`, `tower`, `tower-http`.
- `src/lib.rs` - Library root, re-exports `AppBuilder`/`Config`.
- `src/main.rs` - Binary entrypoint (Axum server).
- `src/error.rs` - API error types.
- `src/openapi.rs` - OpenAPI schema generation.
- `src/rest/` - REST routes, handlers, middleware.
- `src/graphql/` - GraphQL schema and types.
- `src/services/` - Per-context service wiring (audio, embedding, vector, cluster, interpretation).
- `src/websocket/` - WebSocket handlers for streaming progress updates.

## Run
```
cargo run -p sevensense-api
```

## Related
- ADRs covering API design: `../../docs/adr/ADR-008-api-design.md`.
- All other `sevensense-*` crates.
