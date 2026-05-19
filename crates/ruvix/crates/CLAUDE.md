# ruvix/crates

Subsystem crates of the RuVix Cognition Kernel (ADR-087). Each is an `rlib` (mostly `no_std`) library implementing one piece of
the kernel. They are wired together by `ruvix-nucleus` (`nucleus/`) which exposes the 12-syscall dispatch table.

## Layering (rough dependency order)

1. `types` — zero-dep `no_std` kernel interface types (six primitives).
2. `hal` — Hardware Abstraction Layer traits (Console/IRQ/Timer/MMU/Power).
3. `region`, `cap`, `queue` — core kernel objects.
4. `proof` — proof engine consuming `cap`.
5. `sched` — coherence-aware scheduler consuming `cap`.
6. `vecgraph` — kernel-resident vector + graph stores on `types` + `region`.
7. `boot` — RVF boot loader using `region`/`queue`/`cap`.
8. `smp` — symmetric multi-processing primitives on `hal`.
9. `physmem`, `dma`, `dtb` — memory + device discovery on `hal`.
10. `aarch64` — AArch64 arch support (boot.S, MMU, exception vectors).
11. `drivers` — PL011 UART, GICv2, ARM generic timer (uses `hal`).
12. `bcm2711` + `rpi-boot` — Raspberry Pi 4/5 (BCM2711/BCM2712) support.
13. `net`, `fs` — Phase E networking and filesystem.
14. `shell` — in-kernel debug shell.
15. `nucleus` — integration crate (syscall dispatch + deterministic replay + witness log).
16. `cli` — host-side `ruvix` CLI binary (not `no_std`).

Each subdir has its own `CLAUDE.md` with details.
