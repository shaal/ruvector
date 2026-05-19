# ruvector-attention-cli/src/commands

One file per clap subcommand. Each exposes an `Args` struct used in the
parent `Commands` enum and a `run` function invoked from `main.rs`.

## Files
- `mod.rs` - module declarations.
- `compute.rs` - `Compute` subcommand: run attention over an input batch.
- `benchmark.rs` - `Benchmark` subcommand: micro/macro perf measurements.
- `convert.rs` - `Convert` subcommand: format conversions between JSON / MsgPack.
- `serve.rs` - `Serve` subcommand: launches the axum HTTP server in
  `../server`.
- `repl.rs` - interactive REPL using `rustyline`.
