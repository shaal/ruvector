# ui/ruvocal/src/routes/api/v2/user/

v2 user endpoint and sub-resources.

## Files

- `+server.ts` — `GET` current user profile (Pro status, billing summary, settings pointer). Uses `requireAuth`.

## Subdirectories

- `billing-orgs/` — list billing organizations the user belongs to.
- `reports/` — user-submitted moderation reports.
- `settings/` — user settings get/update.
