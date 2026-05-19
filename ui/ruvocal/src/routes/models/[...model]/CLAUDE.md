# ui/ruvocal/src/routes/models/[...model]/

Per-model detail page. The `[...model]` rest-segment captures the full `namespace/model` slug.

## Files

- `+page.svelte` — model detail UI (description, capabilities, "open in chat" button).
- `+page.ts` — load function that fetches the model via `lib/APIClient.ts` → `/api/v2/models/[namespace]/[model]`.

## Subdirectories

- `thumbnail.png/` — dynamic Open Graph image endpoint (renders with satori + the bundled Inter fonts under `lib/server/fonts/`).
