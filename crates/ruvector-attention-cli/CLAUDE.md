# ruvector-attention-cli

Command-line interface and HTTP server for the `ruvector-attention` crate
(high-performance attention mechanisms). Binary name: `ruvector-attention`.

## Important files
- `Cargo.toml` - declares `[[bin]] name = "ruvector-attention"` at
  `src/main.rs`. Deps include `clap`, `tokio (full)`, `axum`, `tower-http`,
  `tracing-subscriber`, `rustyline` (REPL), `indicatif`, `tabled`,
  `rmp-serde` (MessagePack output).
- `src/main.rs` - clap `Parser` entry point and the `Commands` enum
  (`Compute`, `Benchmark`, `Convert`, `Serve`, REPL).
- `src/config.rs` - configuration loading from TOML.
- `src/output.rs` - output formatters (JSON / MessagePack / table via
  `tabled`).
- `src/commands/` - per-subcommand implementations.
- `src/server/` - axum-based HTTP server for the `Serve` subcommand.
- `config/default.toml` - default configuration used when `--config` is
  omitted.

## Subcommands
`Compute`, `Benchmark`, `Convert`, `Serve`, plus an interactive REPL.

## Tests / benches
- `dev-dependencies = criterion = "0.5"` - benches live alongside code, none
  in this dir tree.

## Related
- `../ruvector-attention` - the underlying library (path dep, v2).
- The HTTP `Serve` mode mirrors the attention library API; useful for
  language-agnostic clients.
