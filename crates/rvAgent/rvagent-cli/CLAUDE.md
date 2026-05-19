# rvagent-cli

The `rvagent` terminal coding agent. Wraps `rvagent-core`, `rvagent-backends`,
`rvagent-middleware`, `rvagent-tools`, `rvagent-subagents`, and `rvagent-a2a`
into a single binary with an interactive TUI, single-prompt mode, session
management, and MCP-tool integration.

## Layout

- `Cargo.toml` — bin `rvagent` (`src/main.rs`). Depends on all rvAgent
  subcrates, clap, tokio (signal/process/io-std/...), console.
- `src/main.rs` — clap-based CLI entry; dispatches to `App` / `SessionAction`.
- `src/lib.rs` — placeholder for re-use as a library.
- `src/app.rs` — top-level `App` (agent loop orchestrator).
- `src/tui.rs` — interactive terminal UI.
- `src/display.rs` — rich display helpers (`console`).
- `src/session.rs` — session management (`SessionAction`).
- `src/mcp.rs` — MCP-tool wiring inside the CLI.
- `src/a2a.rs` — A2A peer integration inside the CLI.
- `tests/` — `integration_tests.rs`, `a2a_cli.rs`.

## Related

All other `rvAgent/rvagent-*` crates.
