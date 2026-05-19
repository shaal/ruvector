# @ruvector/postgres-cli

Advanced AI vector-database CLI and SDK for PostgreSQL — positioned as a pgvector drop-in with 53+ SQL functions, 39 attention mechanisms, GNN layers, hyperbolic embeddings, agent routing, sparse vectors, and self-learning features exposed via the RuVector PostgreSQL extension. Ships both a `ruvector-pg` / `rvpg` CLI and a library API.

## Important files

- `package.json` — `@ruvector/postgres-cli` v0.2.7. Main `dist/index.js`, types `dist/index.d.ts`, ESM. Bin: `ruvector-pg` and `rvpg` → `dist/cli.js`. Deps: `pg`, `commander`, `chalk`, `inquirer`, `ora`, `cli-table3`. Scripts: `build`/`dev`/`clean`/`test` (node --test)/`typecheck`/`lint`/`prepublishOnly`.
- `src/index.ts` — library entry. Exports `RuVectorClient` + types, plus command classes (`VectorCommands`, `AttentionCommands`, `GnnCommands`, `GraphCommands`, `LearningCommands`, `BenchmarkCommands`).
- `src/cli.ts` — `commander`-based CLI wiring all command groups (vector, attention, gnn, graph, learning, benchmark, sparse, hyperbolic, routing, quantization, install).
- `src/client.ts` — `RuVectorClient`: pg-Pool-backed PostgreSQL connection with connection pooling, retry+backoff, batched inserts, SQL-injection protection, validation. Defines `PoolConfig`, `RetryConfig`, and result types (`RuVectorInfo`, `VectorSearchResult`, `AttentionResult`, `GnnResult`, `GraphNode`, `GraphEdge`, `TraversalResult`).
- `src/commands/` — per-feature command modules (one file per subcommand group).
- `benchmarks/` — SQL benchmark scripts.
- `tests/` — install/integration scripts.

## Related

- Postgres extension itself lives in the Rust workspace (likely `crates/ruvector-postgres` or similar).
- Sibling CLIs: `npm/packages/rudag` (CLI), `npm/packages/pi-brain` (CLI).
