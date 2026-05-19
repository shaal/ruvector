# ruvix/benches — ruvix-bench

Workspace-level benchmark crate comparing RuVix Cognition Kernel syscalls against equivalent Linux syscalls. Provides four bin
targets (`ruvix-vs-linux`, `syscall-bench`, `proof-overhead`, plus throughput/memory) and a library of comparison/reporting helpers.

## Files

- `Cargo.toml` — `ruvix-bench` package (`publish = false`). Declares `[[bin]]` targets and four `[[bench]]` criterion harnesses
  (`syscall_benches`, `proof_tiers`, `throughput`, `linux_comparison`).
- `benches/` — criterion harnesses (`syscall_benches.rs`, `proof_tiers.rs`, `throughput.rs`, `linux_comparison.rs`).
- `src/` — `lib.rs` plus helpers (`comparison.rs`, `linux.rs`, `report.rs`, `ruvix.rs`, `stats.rs`, `targets.rs`) and `src/bin/`
  containing the binary entry points.

## Subdirs

- `benches/` — see `benches/CLAUDE.md`.
- `src/` — see `src/CLAUDE.md`.
- `src/bin/` — see `src/bin/CLAUDE.md`.

Run: `cargo bench -p ruvix-bench` or `cargo run -p ruvix-bench --bin <name>`.
