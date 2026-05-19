# ruvix — RuVix Cognition Kernel

Standalone Cargo workspace (its own `resolver = "2"` workspace, nested inside the outer `ruvector` repo) implementing the **RuVix
Cognition Kernel** per ADR-087: a seL4-inspired, capability-based, proof-gated, `no_std` microkernel where vector and graph stores
are first-class kernel objects. Targets AArch64 bare-metal (QEMU virt, Raspberry Pi 4/5).

The six core kernel primitives are: **Task**, **Capability**, **Region**, **Queue**, **Timer**, **Proof**.

## Top-level layout

- `Cargo.toml` — workspace root listing 27 member crates and workspace-shared dependency versions for all `ruvix-*` packages.
- `Cargo.lock` — workspace lockfile (kernel needs reproducible builds).
- `aarch64-boot/` — bare-metal boot artifacts (linker script, target spec JSON, Makefile, `build.rs`, `.cargo/`).
- `crates/` — 22 subsystem crates. Each follows ADR-087 phases (Phase A core, B SMP, C IPC/sched, D RPi, E net/fs, etc.).
- `benches/` — `ruvix-bench` package: cross-cutting benchmarks comparing RuVix vs Linux syscalls.
- `tests/` — `ruvix-integration` package: integration tests for ADR-087 Section 17 acceptance criteria.
- `examples/` — `cognitive_demo` (`ruvix-demo`) and `rvf-demos/swarm-consensus` showcasing full kernel use.
- `qemu-swarm/` — `ruvix-qemu-swarm`: QEMU-based distributed cluster simulation harness.

## Subsystem crate map (under `crates/`)

| Crate | Role |
|---|---|
| `types` | Zero-dep `no_std` kernel interface types (six primitives) |
| `region` | Memory region manager: Immutable / AppendOnly / Slab |
| `queue` | io_uring-style ring buffer IPC (zero-copy) |
| `cap` | seL4-style capability manager + derivation tree |
| `proof` | 3-tier proof engine (Reflex / Standard / Deep) |
| `sched` | Coherence-aware EDF scheduler with novelty/risk signals |
| `boot` | RVF boot loading + ML-DSA-65 signature verification |
| `vecgraph` | Kernel-resident vector + graph stores (HNSW) |
| `nucleus` | Integration crate: 12-syscall dispatch + deterministic replay |
| `hal` | Hardware Abstraction Layer (Console/IRQ/Timer/MMU/Power) |
| `aarch64` | AArch64 arch support (boot.S, MMU, exceptions) |
| `drivers` | PL011 UART, GICv2, ARM generic timer |
| `smp` | SMP primitives: per-CPU, spinlocks, IPI, memory barriers |
| `physmem` | Buddy allocator for physical page frames |
| `dma` | DMA controller abstraction (scatter-gather, cache coherent) |
| `dtb` | Zero-copy FDT/Device Tree parser |
| `net` | Minimal `no_std` network stack (eth/ARP/IPv4/ICMP/UDP) |
| `fs` | VFS + FAT32 (read-only) + RamFS |
| `bcm2711` | BCM2711/BCM2712 SoC drivers for Raspberry Pi 4/5 |
| `rpi-boot` | Raspberry Pi-specific boot (kernel8.img, DTB handoff) |
| `shell` | In-kernel debug shell (`info`/`mem`/`tasks`/`caps`/...) |
| `cli` | Host-side `ruvix` CLI (build/flash/keys/monitor/security) |

## Building

- Bare-metal AArch64: see `aarch64-boot/Makefile`. Target spec is `aarch64-ruvix.json`.
- Host tooling and tests build normally against the outer Cargo workspace's local copy of this nested workspace.

## Related

- `../ruvector-coherence` — optional integration consumed by `ruvix-sched` via the `coherence` feature.
- The outer `ruvector` repo provides higher-level vector-DB / agent runtimes that can target RuVix.
