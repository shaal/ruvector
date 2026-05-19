# ui/ruvocal/src/lib/server/api/__tests__/

Integration tests for the v2 HTTP API (`src/routes/api/v2/...`).

## Files

- `testHelpers.ts` — shared utilities (in-memory Mongo, fake users, request helpers).
- `conversations.spec.ts` — list/create conversations.
- `conversations-id.spec.ts` — single-conversation get/update/delete.
- `conversations-message.spec.ts` — message create/edit/branch flows.
- `user.spec.ts` — `/api/v2/user` endpoints.
- `user-reports.spec.ts` — `/api/v2/user/reports` endpoints.
- `misc.spec.ts` — miscellaneous endpoint coverage (feature flags, public-config, etc.).

Run via vitest with `scripts/setups/vitest-setup-server.ts`.
