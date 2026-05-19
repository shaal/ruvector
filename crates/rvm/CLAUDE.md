# rvm — RuVix Virtual Machine (RVM)

Coherence-native microhypervisor for edge computing and multi-agent systems. **This directory is its own Cargo workspace** (`resolver = "2"`, separate `Cargo.lock`), not just a crate. It targets `aarch64-unknown-none` by default and runs in QEMU `virt` (`make run`).

## Workspace members

| Crate | Purpose | Key ADRs |
|-------|---------|----------|
| `crates/rvm-types` | Foundation types (`PartitionId`, `Capability`, `WitnessRecord`, `MemoryRegion`, `CommEdge`, ...) | ADR-132/133/134 |
| `crates/rvm-hal` | Hardware abstraction layer (`Platform`, `MmuOps`, `TimerOps`, `InterruptOps`) | ADR-133 |
| `crates/rvm-cap` | Capability system with P1/P2 proof verification | ADR-135 |
| `crates/rvm-witness` | Witness logging (64-byte records, FNV-1a hash chain) | ADR-134 |
| `crates/rvm-proof` | Proof-gated state transitions (Hash / Witness / Zk tiers; TEE attestation) | ADR-135, ADR-142 |
| `crates/rvm-partition` | Partition object model and lifecycle (split / merge / migrate) | ADR-133 |
| `crates/rvm-sched` | Coherence-aware scheduler (`deadline_urgency + cut_pressure_boost`) | ADR-132 DC-4 |
| `crates/rvm-memory` | Four-tier (Hot/Warm/Dormant/Cold) memory manager | ADR-136, ADR-138 |
| `crates/rvm-coherence` | Coherence scoring + Phi computation | ADR-139 |
| `crates/rvm-boot` | 7-phase deterministic boot sequence | ADR-137, ADR-140 |
| `crates/rvm-wasm` | Optional WebAssembly guest runtime | ADR-140 |
| `crates/rvm-security` | Unified security gate (capability + proof + witness) | — |
| `crates/rvm-kernel` | Top-level integration; emits the `rvm` binary | — |
| `tests/` | Cross-crate integration tests | — |
| `benches/` | Criterion benchmarks (coherence / witness / overall) | — |

## Layout

- `Cargo.toml` — workspace manifest; `[workspace.dependencies]` wires internal path deps and shared external deps (`spin`, `bitflags`, `subtle`, `sha2`, `hmac`, `ed25519-dalek`, `criterion`).
- `Cargo.lock` — committed (separate workspace).
- `rvm.ld` — linker script for the AArch64 image.
- `Makefile` — `build`, `check`, `run` (QEMU `cortex-a72`, 128 M), `test`, `clean`. Target `aarch64-unknown-none`.
- `.cargo/`, `.github/` — local cargo config and CI.

## How it fits the parent ruvector tree

The `rvm` workspace is committed inside `crates/` so the parent workspace can compose it, but builds and lints separately. The kernel and partition model echo the same coherence / mincut / witness language used elsewhere in ruvector (see `crates/cognitum-gate-kernel`, `crates/neural-trader-coherence`).
