# ruvector-cli/src/cli

CLI subcommand implementations and helpers.

- `mod.rs` — module roots; re-exports everything for `main.rs`.
- `commands.rs` — concrete handlers for `ruvector` subcommands.
- `format.rs` — output formatting (`prettytable`, colored).
- `graph.rs` — graph inspection / dump helpers using `ruvector-graph`.
- `hooks.rs` — in-memory hooks engine (event-driven side-effects).
- `hooks_postgres.rs` — Postgres-backed hooks (feature = "postgres").
  Uses `tokio-postgres` + `deadpool-postgres`; schema in `../../sql/hooks_schema.sql`.
- `progress.rs` — `ProgressTracker` wrapping `indicatif`.
