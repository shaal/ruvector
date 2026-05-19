# sevensense-api/src

Source for the 7sense REST/GraphQL/WebSocket server.

## Files
- `lib.rs` - Library root. Documents the layered architecture (REST `/api/v1/*`, GraphQL `/graphql`, WS `/ws`) and exports `AppBuilder`/`Config`.
- `main.rs` - Binary entrypoint launching the Axum server.
- `error.rs` - API-level error type and HTTP mapping.
- `openapi.rs` - OpenAPI schema generation for the REST endpoints.

## Subdirectories
- `rest/` - REST routes, handlers, middleware.
- `graphql/` - GraphQL schema and types.
- `services/` - Per-domain service wiring (audio, cluster, embedding, interpretation, vector).
- `websocket/` - WebSocket handlers (real-time updates).

## Related
- Parent: `../CLAUDE.md`.
