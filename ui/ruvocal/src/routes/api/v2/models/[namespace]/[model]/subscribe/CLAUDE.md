# ui/ruvocal/src/routes/api/v2/models/[namespace]/[model]/subscribe/

Server-Sent Events subscription for a single model.

## Files

- `+server.ts` — `GET` opens an SSE stream pushing status/metadata changes for `[namespace]/[model]` (online/offline, quota changes). Client consumers update the model picker live.
