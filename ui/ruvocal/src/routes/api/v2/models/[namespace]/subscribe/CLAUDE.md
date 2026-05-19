# ui/ruvocal/src/routes/api/v2/models/[namespace]/subscribe/

Server-Sent Events subscription for an entire model namespace.

## Files

- `+server.ts` — `GET` opens an SSE stream that emits add/remove/status events for any model under `[namespace]`. Cheaper than subscribing per-model when you only care about a provider.
