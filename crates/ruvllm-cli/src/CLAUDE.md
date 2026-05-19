# ruvllm-cli/src

CLI source. `main.rs` declares the clap parser and dispatches into
`commands/`.

## Files
- `main.rs` - clap `Cli` + `Commands` enum (Download, List, Info, Serve,
  Chat, Benchmark, Quantize). Global flags: `--verbose`, `--no-color`,
  `--cache-dir` (env `RUVLLM_CACHE_DIR`).
- `models.rs` - model identifier resolution (HF IDs, aliases like `qwen`,
  `mistral`, `phi`, `llama`) and cache-path helpers.
- `commands/` - per-subcommand implementations.
