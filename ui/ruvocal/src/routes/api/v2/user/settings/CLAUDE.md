# ui/ruvocal/src/routes/api/v2/user/settings/

User settings get/update.

## Files

- `+server.ts` — `GET` returns the user's stored settings (`lib/types/Settings.ts`); `PATCH` updates them (theme, default model, system prompt, MCP servers, etc.). Counterpart to the `lib/stores/settings.ts` store.
