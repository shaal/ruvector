# ruvbot / deploy

Deployment assets for shipping `ruvbot` to managed infrastructure.

## Files
- `init-db.sql` - PostgreSQL schema bootstrap (tenant, session, memory
  tables) used by the optional `pg` persistence backend (ADR-003).
- `gcp/` - Google Cloud Run + Cloud Build assets (see subdir).

Use these in combination with the top-level `Dockerfile` and
`docker-compose.yml` for self-hosting.
