# ui/ruvocal/src/routes/admin/

Admin-only routes. Protected by `lib/server/adminToken.ts` — callers must present an admin token.

## Subdirectories

- `export/` — bulk data export.
- `stats/` — operational stats (with a `compute/` subroute that triggers recomputation via `lib/jobs/refresh-conversation-stats.ts`).
