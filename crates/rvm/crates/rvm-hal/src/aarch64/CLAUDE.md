# rvm-hal/src/aarch64

Concrete AArch64 implementation of the HAL traits. Targets the QEMU `virt` machine with `cortex-a72` (see `../../../Makefile`).

- `mod.rs` — module wiring; provides the AArch64 `Platform` implementation.
- `boot.rs` — early boot / EL2 entry helpers.
- `mmu.rs` — stage-2 page-table programming (`MmuOps` impl).
- `timer.rs` — generic timer (`TimerOps` impl).
- `interrupts.rs` — GIC routing / masking (`InterruptOps` impl).
- `uart.rs` — early UART for boot-time logging.

`unsafe` is allowed here (register access, MMIO, inline asm). Every `unsafe` block carries a `// SAFETY:` comment. See `../CLAUDE.md`.
