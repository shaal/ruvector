# ui/ruvocal/src/routes/api/v2/feature-flags/

v2 endpoint exposing per-user feature flags.

## Files

- `+server.ts` — `GET` returns the resolved feature-flag map for the authenticated user (based on env config, user pro status, A/B buckets). Consumed by the client during bootstrap (see `lib/utils/PublicConfig.svelte.ts`).
