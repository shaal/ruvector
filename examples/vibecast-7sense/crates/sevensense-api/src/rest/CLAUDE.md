# sevensense-api/src/rest

REST API layer (`/api/v1/*`).

## Files
- `mod.rs` - Module exports.
- `routes.rs` - Route table (Axum `Router`).
- `handlers.rs` - Request handlers (audio upload, search, clusters, evidence packs).
- `middleware.rs` - Auth / tracing / rate-limit middleware.

## Related
- Services delegated to: `../services/`.
- GraphQL counterpart: `../graphql/`.
