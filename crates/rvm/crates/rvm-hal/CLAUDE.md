# rvm-hal

Platform-agnostic Hardware Abstraction Layer for the RVM microhypervisor (ADR-133). Concrete implementations live in per-arch submodules (`aarch64`; RISC-V and x86-64 planned).

## Subsystems / traits

- `Platform` — top-level platform discovery and initialisation.
- `MmuOps` — stage-2 page-table management.
- `TimerOps` — monotonic timer and deadline scheduling.
- `InterruptOps` — interrupt routing and masking.

## Design constraints (ADR-133)

All trait methods return `RvmResult`. No `unsafe` in trait *definitions*; impls may need it (annotated with `// SAFETY:`). Zero-copy: borrowed slices only.

`#![no_std] #![deny(unsafe_code)] #![deny(missing_docs)]` (deny, not forbid — concrete arch impls require `unsafe`).

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`.
- `src/lib.rs` — trait definitions and public re-exports.
- `src/aarch64/` — AArch64 implementation (`boot`, `interrupts`, `mmu`, `timer`, `uart`).

See `../CLAUDE.md`.
