# ui/ruvocal/src/routes/settings/(nav)/[...model]/

Per-model settings page. Rest-segment captures the full `namespace/model` slug.

## Files

- `+page.svelte` — UI to edit per-model overrides (system prompt, sampling params, default for this user, etc.).
- `+page.ts` — load: resolves the model via `lib/APIClient.ts` → `/api/v2/models/[namespace]/[model]`.
