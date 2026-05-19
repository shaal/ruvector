# ruvix-cli — `ruvix` host-side CLI

Host-side tooling for building, flashing, configuring, and monitoring RuVix kernel images on AArch64 bare-metal targets. Single
`[[bin]]` named `ruvix` (path `src/main.rs`). Supports secure boot, key management, DTB validation, and serial monitoring.

## Files

- `Cargo.toml` — single binary `ruvix`. Uses workspace-shared `clap`, serde, anyhow.
- `src/main.rs` — Clap-derived CLI dispatching to subcommands.
- `src/commands/` — one module per subcommand (see `src/commands/CLAUDE.md`).

## Example invocations

- `ruvix build --release --secure-boot --target aarch64-unknown-none`
- `ruvix flash --device /dev/sdb --image target/kernel8.img`
- `ruvix keys generate ...`
