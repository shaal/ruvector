# ruvector-cli/src

Top-level source of the CLI and MCP binaries.

- `main.rs` — `ruvector` CLI entry. Defines clap `Cli` and `Commands` enums
  (Create, Insert, Search, ...), dispatches to `cli::commands`.
- `mcp_server.rs` — `ruvector-mcp` server entry.
- `config.rs` — `Config` struct, TOML loader.
- `cli/` — subcommand implementations (`commands.rs`), output formatting, graph
  inspection, hook engine (in-memory + optional Postgres), progress UI.
- `mcp/` — MCP protocol types, JSON-RPC handlers, stdio/SSE transports, GNN LRU
  cache.
