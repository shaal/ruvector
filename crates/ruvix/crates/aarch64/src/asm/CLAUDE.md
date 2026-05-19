# ruvix-aarch64/src/asm

Hand-written AArch64 assembly included by the parent crate via `core::arch::global_asm!` / `include_str!`.

## Files

- `boot.S` — `_start` entry point. Disables interrupts, sets up the stack pointer, clears BSS, calls `early_init()`, configures
  exception vectors, then jumps to `kernel_main()`.
- `vectors.S` — exception vector table (sync / IRQ / FIQ / SError, per-EL).
