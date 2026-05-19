# ui/ruvocal/src/routes/api/v2/export/

v2 endpoint for user data export (self-service, not admin).

## Files

- `+server.ts` — `GET` streams the authenticated user's conversations/data as a downloadable archive (parquet / zip via `yazl`).
