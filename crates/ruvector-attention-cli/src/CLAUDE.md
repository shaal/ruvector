# ruvector-attention-cli/src

CLI source. Entry point is `main.rs`, which dispatches into `commands/` and
optionally serves via `server/`.

## Files
- `main.rs` - clap `Cli` + `Commands` enum, tracing init, dispatch.
- `config.rs` - load/merge TOML config (defaults from `../config/default.toml`).
- `output.rs` - output formatters (table via `tabled`, JSON via `serde_json`,
  MessagePack via `rmp-serde`).
- `commands/` - subcommand implementations (compute, benchmark, convert,
  serve, repl).
- `server/` - axum HTTP server for the `Serve` subcommand.
