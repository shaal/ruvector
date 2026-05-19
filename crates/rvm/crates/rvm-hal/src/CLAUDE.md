# rvm-hal/src

- `lib.rs` — crate root. Defines the four HAL traits (`Platform`, `MmuOps`, `TimerOps`, `InterruptOps`) and re-exports the concrete arch implementations.
- `aarch64/` — AArch64 concrete impl; see `aarch64/CLAUDE.md`.

See `../CLAUDE.md`.
