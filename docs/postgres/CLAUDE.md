# docs/postgres/

PostgreSQL extension docs for ruvector. Covers `sparsevec` (pgvector-compatible sparse vectors), parallel query support, and zero-copy memory integration. Audience: integrators using ruvector inside Postgres.

## Top-level docs

- `SPARSEVEC_QUICKSTART.md` - sparsevec quick start.
- `SPARSEVEC_IMPLEMENTATION.md` - implementation notes.
- `operator-quick-reference.md` - SQL operator reference.
- `parallel-query-guide.md` - parallel query usage.
- `parallel-implementation-summary.md` - parallel implementation summary.
- `postgres-memory-implementation-summary.md` - in-Postgres memory implementation.
- `postgres-zero-copy-memory.md` - zero-copy memory design.
- `postgres-zero-copy-quick-reference.md` - zero-copy quick reference.

## Subdirs

- `v2/` - v2 redesign: SQL schema, background workers, index AMs, integrity, tiered storage, Cypher, hybrid search, multi-tenancy, self-healing.
- `zero-copy/` - zero-copy operator implementation details + example code.

## Related

- `../adr/ADR-044-ruvector-postgres-v03-extension-upgrade.md` - upgrade ADR.
- `../sql/`, `../examples/sparsevec_examples.sql` - SQL examples.
- `../dag/04-POSTGRES-INTEGRATION.md` - DAG executor integration with Postgres.
