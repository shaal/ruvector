# ruvix-aarch64

AArch64 architecture support for the RuVix Cognition Kernel: boot sequence + early init, MMU configuration, exception handling
(sync/IRQ/FIQ/SError), and system register access. Falls back to an empty shell when built for non-AArch64 targets so workspace-
wide `cargo build` succeeds.

## Memory layout

```
0x0000_0000_0000_0000 - 0x0000_FFFF_FFFF_FFFF  User space (TTBR0_EL1)
0xFFFF_0000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF  Kernel space (TTBR1_EL1)
```

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-hal`. Feature: `qemu-virt`.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
