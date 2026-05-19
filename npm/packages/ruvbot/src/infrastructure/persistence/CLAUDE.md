# ruvbot / src / infrastructure / persistence

Persistence layer (ADR-003). Defines repository interfaces and the
Postgres-backed implementation used for tenants, sessions, agents,
messages, and memory rows. Falls back to in-memory stores when `pg`
is not installed.

## Files
- `index.ts` - Barrel exposing repository contracts and concrete
  implementations consumed by the core context.

Schema bootstrap lives in `../../../deploy/init-db.sql`.
