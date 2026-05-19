# ruvector-cli

High-performance Rust vector database CLI plus an MCP server. Provides commands for
create / insert / search / stats / etc. over `ruvector-core` + `ruvector-graph` +
`ruvector-gnn`, and exposes the database through the Model Context Protocol so it can
be embedded into Claude / IDE workflows. Optional Postgres-backed hooks system.

## Layout

- `Cargo.toml` — two bins: `ruvector` (`src/main.rs`) and `ruvector-mcp`
  (`src/mcp_server.rs`). Feature `postgres = [tokio-postgres, deadpool-postgres]`.
  Deps: clap, tokio, axum/hyper (for SSE MCP transport), tower, lru, flate2,
  ndarray + ndarray-npy, csv, prettytable-rs, colored, indicatif.
- `src/main.rs` — `ruvector` CLI entry: subcommands defined via clap (Create, Insert,
  Search, etc.), routes to `cli::commands`.
- `src/mcp_server.rs` — `ruvector-mcp` server binary entry.
- `src/config.rs` — `Config` (TOML loader).
- `src/cli/` — command implementations and TTY helpers (see CLAUDE.md).
- `src/mcp/` — JSON-RPC MCP protocol, transport, and tool handlers (see CLAUDE.md).
- `tests/` — integration tests: CLI, MCP, hooks, GNN performance.
- `docs/IMPLEMENTATION.md` — design / implementation notes.
- `scripts/` — bash + PowerShell statusline scripts.
- `sql/hooks_schema.sql` — Postgres schema for the hooks subsystem.
- `.claude/` — Claude Code workspace config.

## Related

- `crates/ruvector-core`, `crates/ruvector-graph`, `crates/ruvector-gnn` — engines.
