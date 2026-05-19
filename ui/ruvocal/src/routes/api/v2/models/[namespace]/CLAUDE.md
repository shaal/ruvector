# ui/ruvocal/src/routes/api/v2/models/[namespace]/

v2 namespace-level model routes.

## Files

- `+server.ts` — `GET` list of models within `[namespace]`.

## Subdirectories

- `[model]/` — single-model detail (+ nested `subscribe/`).
- `subscribe/` — subscribe to updates for the entire namespace (SSE stream).
