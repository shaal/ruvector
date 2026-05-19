# ruvix/aarch64-boot

Bare-metal boot artifacts for building the RuVix kernel image for AArch64. Not a Cargo crate itself — just the linker script,
custom target spec, Makefile, and a tiny `build.rs` used by the kernel binary build.

## Files

- `Makefile` — convenience targets to build/link the kernel image (`kernel8.img` / ELF) using the custom target.
- `aarch64-ruvix.json` — custom rustc target specification (LLVM triple, code model, features) for the RuVix kernel.
- `linker.ld` — linker script defining the memory layout (text, rodata, bss, stack) and entry point.
- `build.rs` — small build script that points rustc at `linker.ld` (114 bytes).
- `.cargo/` — local Cargo config (target / runner / rustflags overrides for bare-metal builds).

## Notes

- Used in conjunction with `ruvix-aarch64` (low-level arch code in `crates/aarch64`) and `ruvix-nucleus` (top-level kernel entry).
- Run `make` here to produce an image bootable in QEMU virt or on a Raspberry Pi (via `ruvix-rpi-boot`).
