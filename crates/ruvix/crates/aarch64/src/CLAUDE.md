# ruvix-aarch64/src

AArch64-specific code (inline asm, MMU, system registers) — only meaningful on `target_arch = "aarch64"`.

## Files

- `lib.rs` — crate root; conditionally compiled to an empty shell on non-aarch64 targets.
- `boot.rs` — early Rust-side boot path called from `_start` (after `asm/boot.S`).
- `mmu.rs` — MMU configuration (TTBR0/TTBR1, translation tables, attributes).
- `exception.rs` — exception handling (sync, IRQ, FIQ, SError).
- `registers.rs` — system register access wrappers.
- `asm/` — see `asm/CLAUDE.md`.
