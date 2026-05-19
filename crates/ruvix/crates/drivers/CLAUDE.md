# ruvix-drivers

Device drivers for the RuVix Cognition Kernel (ADR-087), designed for the QEMU `virt` machine on AArch64. Implements the
`ruvix-hal` traits with concrete hardware.

## Supported devices

- **PL011 UART** — ARM PrimeCell UART for serial console I/O (0x0900_0000 on QEMU virt).
- **GICv2** — ARM Generic Interrupt Controller, GIC-400 (Distributor 0x0800_0000, CPU IF 0x0800_1000).
- **ARM Generic Timer** — system timer with deadline scheduling.

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-hal`. Feature: `qemu-virt`.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
