# rvf-cli

Unified `rvf` command-line interface for RuVector Format stores. Single binary built from `src/main.rs`.

## Layout

- `Cargo.toml` — `[[bin]] name = "rvf"`. Deps: `rvf-runtime`, `rvf-types`, `rvf-wire`, `rvf-manifest`, `rvf-crypto`; optional `rvf-server`; plus `clap` (derive).
- `src/main.rs` — clap `Cli` enum dispatching to per-command modules in `cmd/`.
- `src/output.rs` — shared output formatting helpers.
- `src/cmd/` — one file per subcommand; see `src/cmd/CLAUDE.md`.

## Subcommands

`create`, `ingest`, `query`, `delete`, `status`, `compact`, `derive`, `embed-ebpf`, `embed-kernel`, `filter`, `freeze`, `inspect`, `launch`, `rebuild-refcounts`, `serve`, `verify-attestation`, `verify-witness`.

## Related

- `../rvf-runtime`, `../rvf-types`, `../rvf-wire`, `../rvf-manifest`, `../rvf-crypto`, `../rvf-server`
- See `../rvf-import` for the dedicated importer binary
