# sevensense-api/src/websocket

WebSocket layer for real-time updates (long-running ingestion, batched embeddings, clustering progress).

## Files
- `mod.rs` - Module wiring + router integration.
- `handlers.rs` - WebSocket upgrade handlers that subscribe to domain events from the services and stream them to clients.

## Related
- Domain event sources: services in `../services/`.
- REST counterpart: `../rest/`.
