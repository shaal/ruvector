# ui/ruvocal/src/routes/settings/(nav)/application/

Application-level settings (`/settings/application`).

## Files

- `+page.svelte` — theme, locale, MCP servers entry, autopilot toggle, advanced flags. Pulls/pushes via `lib/stores/settings.ts` + `/api/v2/user/settings`.
