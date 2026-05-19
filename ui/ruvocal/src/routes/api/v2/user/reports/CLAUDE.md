# ui/ruvocal/src/routes/api/v2/user/reports/

User-side moderation report endpoint.

## Files

- `+server.ts` — `GET`/`POST` reports filed by the authenticated user (backed by `lib/types/Report.ts`). Tested in `lib/server/api/__tests__/user-reports.spec.ts`.
