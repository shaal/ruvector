# rvf-cli/src

Source for the `rvf` binary.

## Files

- `main.rs` — clap `Cli` + `Commands` enum, error handling, top-level dispatch.
- `output.rs` — shared output formatting helpers (tables, JSON, status lines).
- `cmd/` — one file per subcommand. See `cmd/CLAUDE.md`.
