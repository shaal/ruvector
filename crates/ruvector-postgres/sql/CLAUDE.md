# ruvector-postgres/sql

PostgreSQL extension SQL scripts (versioned + per-feature examples).

## Extension scripts (loaded by `CREATE EXTENSION`)

- `ruvector--0.1.0.sql` — Initial release.
- `ruvector--0.3.0.sql` — 0.3 release script.
- `ruvector--2.0.0.sql` — 2.0 release script.
- `ruvector--2.0.0--0.3.0.sql` — Downgrade path from 2.0 to 0.3.

## Feature examples

- `access_methods.sql` — `ruhnsw` / `ruivfflat` usage.
- `hnsw_index.sql` — HNSW examples.
- `ivfflat_am.sql` — IVFFlat AM examples.
- `embeddings.sql` — Local embedding generation.
- `graph_examples.sql` — Graph storage + Cypher.
- `routing_example.sql` — Tiny Dancer routing.

## Pointers

- Extension control file: `../ruvector.control`.
