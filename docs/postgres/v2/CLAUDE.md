# docs/postgres/v2/

The **v2 redesign** of the ruvector PostgreSQL extension. Numbered docs read as an ordered series; covers schema, workers, AMs, integrity, phased compatibility migration, replication, hybrid search, multi-tenancy, and self-healing.

## Docs

- `00-overview.md` - v2 overview.
- `01-sql-schema.md` - SQL schema.
- `02-background-workers.md` - background worker design.
- `03-index-access-methods.md` - custom index access methods.
- `04-integrity-events.md` - integrity event log.
- `05-phase1-pgvector-compat.md` - phase 1: pgvector compatibility.
- `06-phase2-tiered-storage.md` - phase 2: tiered storage.
- `07-phase3-graph-cypher.md` - phase 3: graph + Cypher.
- `08-phase4-integrity-control.md` - phase 4: integrity control.
- `09-migration-guide.md` - migration guide.
- `10-consistency-replication.md` - consistency and replication.
- `11-hybrid-search.md` - hybrid search.
- `12-multi-tenancy.md` - multi-tenancy.
- `13-self-healing.md` - self-healing.

## Related

- `../` - parent postgres docs (v1 era + sparsevec).
- `../zero-copy/` - zero-copy operator details.
- `../../adr/ADR-044-ruvector-postgres-v03-extension-upgrade.md`.
