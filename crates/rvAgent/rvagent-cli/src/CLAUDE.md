# rvagent-cli/src

Source for the `rvagent` terminal binary.

- `main.rs` — clap-based entry, tracing init, dispatch to `App` /
  `SessionAction`.
- `lib.rs` — placeholder for library re-use.
- `app.rs` — `App` orchestrator (agent loop, model/backends/middleware wiring).
- `tui.rs` — interactive terminal UI (TUI).
- `display.rs` — rich display helpers using `console`.
- `session.rs` — `SessionAction` enum and session lifecycle.
- `mcp.rs` — MCP server/client integration inside the CLI.
- `a2a.rs` — A2A peer integration inside the CLI.
