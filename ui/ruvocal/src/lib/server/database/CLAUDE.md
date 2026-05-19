# ui/ruvocal/src/lib/server/database/

Database clients beyond the main MongoDB module (`../database.ts`).

## Files

- `postgres.ts` — `pg` Postgres pool/client used for relational data (analytics / billing).
- `rvf.ts` — client for the **rvf** (Rvector) backing service / KV store referenced from `../../../rvf.manifest.json`.

## Subdirectories

- `__tests__/` — tests for the Postgres / rvf clients.
