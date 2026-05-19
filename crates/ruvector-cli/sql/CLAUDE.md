# ruvector-cli/sql

SQL schemas used by the optional Postgres-backed subsystem.

- `hooks_schema.sql` — DDL for the hooks tables consumed by
  `src/cli/hooks_postgres.rs` (only enabled with feature `postgres`).
