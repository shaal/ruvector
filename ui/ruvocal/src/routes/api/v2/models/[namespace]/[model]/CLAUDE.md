# ui/ruvocal/src/routes/api/v2/models/[namespace]/[model]/

v2 single-model endpoint.

## Files

- `+server.ts` — `GET` returns detailed metadata for `[namespace]/[model]` (capabilities, params, multimodal flags, tool support). Resolved via `lib/server/api/utils/resolveModel.ts`.

## Subdirectories

- `subscribe/` — SSE subscription for live model-status changes.
