# ui/ruvocal/src/routes/settings/(nav)/

SvelteKit route group `(nav)` — does not add a URL segment, but provides the settings-sidebar nav layout for all child routes.

## Files

- `+layout.svelte` — settings sidebar + content frame.
- `+layout.ts` — load function that fetches user/settings via `lib/APIClient.ts`.
- `+page.svelte` — landing settings page (`/settings`).
- `+server.ts` — endpoint actions on the settings landing (e.g. PATCH defaults).

## Subdirectories

- `[...model]/` — per-model settings (model overrides, default params).
- `application/` — application-level settings (theme, locale, advanced toggles).
