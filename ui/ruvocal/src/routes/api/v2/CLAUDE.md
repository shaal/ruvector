# ui/ruvocal/src/routes/api/v2/

The **v2 HTTP API** — newer surface backed by `lib/server/api/` helpers (composable `requireAuth` / `resolveConversation` / `resolveModel` + `superjsonResponse`). Client uses `lib/APIClient.ts` to call these.

## Subdirectories

- `conversations/` — list/create + `[id]/` for single conversation, including `message/[messageId]/` and `import-share`.
- `export/` — user-facing data export.
- `feature-flags/` — runtime feature flags.
- `models/` — namespaced model catalog with subscribe streams.
- `public-config/` — public (non-secret) client config bootstrap.
- `user/` — current user, billing-orgs, reports, settings.

## Conventions

- Endpoints serialize with superjson so Date/Map/Set/BigInt round-trip.
- New endpoints should live under `v2/`, not `../`.
- Tests in `lib/server/api/__tests__/`.
