# ui/ruvocal/src/routes/api/v2/public-config/

Public client-config bootstrap.

## Files

- `+server.ts` — `GET` returns the non-secret config the client needs at startup (branding, feature toggles, version, public sep token). Consumed by `lib/utils/PublicConfig.svelte.ts`.
