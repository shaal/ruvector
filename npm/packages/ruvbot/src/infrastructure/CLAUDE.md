# ruvbot / src / infrastructure

Infrastructure bounded context. Provides the ports/adapters used by
the core domain: persistence, messaging/event bus, and background
worker pools.

## Files
- `index.ts` - Barrel re-exporting the three submodules below.

## Subdirectories
- `persistence/` - Postgres / in-memory repositories (ADR-003).
- `messaging/` - Domain event bus and queue manager (ADR-004).
- `workers/` - Worker pool abstraction used by background jobs.
