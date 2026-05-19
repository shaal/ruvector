# ruvix-hal

Hardware Abstraction Layer for the RuVix Cognition Kernel (ADR-087). Defines the platform-agnostic trait surface every
arch/SoC must implement. `#![forbid(unsafe_code)]` in trait definitions, `#![no_std]`, zero-copy where possible, Result-based.

## Five subsystems

- **Console** — serial I/O for debugging / logging.
- **InterruptController** — IRQ/FIQ management + routing.
- **Timer** — monotonic time + deadline scheduling.
- **MMU** — virtual memory + page tables.
- **PowerManagement** — CPU power states + reset control.

## Files

- `Cargo.toml` — depends only on `ruvix-types`. Pure-traits crate.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.

## Implementers

- `../aarch64` — AArch64 implementation.
- `../drivers` — concrete device drivers fulfilling the traits (PL011, GICv2, ARM generic timer).
- `../bcm2711` — BCM2711/BCM2712 SoC.
