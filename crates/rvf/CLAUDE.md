# rvf — RuVector Format

Independent Cargo workspace inside the ruvector monorepo implementing the RuVector Format (RVF): an append-only, segment-based, cryptographically witnessed vector-store file format with progressive HNSW indexing, temperature-tiered quantization, Linux microkernel embedding for cognitive containers, and a federated transfer-learning protocol.

This directory has its own `Cargo.toml` (a `[workspace]` listing every `rvf-*` member) and `Cargo.lock`. Member crates live in flat siblings and under `rvf-adapters/` and `tests/`.

## Workspace layout

Core / format crates:
- `rvf-types/` — segment headers, enums, flags, error codes; `no_std` baseline.
- `rvf-wire/` — zero-copy wire-format reader/writer + per-segment codecs.
- `rvf-manifest/` — two-level (L0 fixed-size hotset + L1 TLV directory) manifest for progressive boot.
- `rvf-index/` — progressive HNSW (Layer A/B/C tiered search).
- `rvf-quant/` — temperature-tiered quantization (f32/f16/u8/binary + PQ + count-min sketch).
- `rvf-crypto/` — SHAKE-256 hashing, Ed25519 signing, attestation, witness chains, lineage.
- `rvf-runtime/` — `RvfStore` user-facing API: writes, queries, compaction, CoW, AGI containers, QR seed bootstrap.

Surface / runtime crates:
- `rvf-server/` — TCP/HTTP/WebSocket server over `rvf-runtime`.
- `rvf-node/` — Node.js NAPI bindings (`cdylib`) with prebuilt platform binaries under `rvf-node/npm/`.
- `rvf-wasm/` — Cognitum-tile WASM microkernel (no_std, <8 KB after wasm-opt).
- `rvf-solver-wasm/` — self-learning temporal solver WASM (Thompson Sampling + three-loop architecture).
- `rvf-cli/` — unified `rvf` CLI (create/ingest/query/compact/launch/serve/inspect/freeze/verify/...).
- `rvf-import/` — JSON / CSV / NumPy importer + `rvf-import` binary.
- `rvf-kernel/` — Linux bzImage/ELF + initramfs builder for embedding kernels into RVF (cognitive containers).
- `rvf-launch/` — QEMU microVM launcher driven from KERNEL_SEG.
- `rvf-ebpf/` — eBPF C sources (XDP distance, socket filter, TC routing) + `clang` compiler shim.
- `rvf-federation/` — PII stripping, differential privacy, FedAvg/FedProx aggregation.

Adapters (`rvf-adapters/`):
- `agentdb/` — agentdb memory ↔ RVF.
- `agentic-flow/` — swarm coordination, learning patterns, witness consensus ↔ RVF.
- `claude-flow/` — claude-flow memory subsystem ↔ RVF with WITNESS_SEG audit.
- `ospipe/` — observation-state pipeline ↔ RVF.
- `rvlite/` — minimal embedded vector-store API over RVF Core Profile.
- `sona/` — SONA trajectories/patterns/experience-replay ↔ RVF.

Tests + benches:
- `tests/rvf-integration/` — workspace-wide acceptance & integration tests.
- `benches/` — Criterion suite (`benches/benches/rvf_benchmarks.rs`).
- `docs/adr/`, `docs/security/` — design records and security audits.

## Where to start

- User-facing API: `rvf-runtime/src/lib.rs` (`RvfStore`).
- File format reference: `rvf-types/src/segment.rs` + `rvf-wire/`.
- CLI for hands-on exploration: `rvf-cli/src/main.rs`.

## Related (outside this workspace)

- `../ruvector-domain-expansion` uses `rvf-types`/`rvf-wire`/`rvf-crypto` under its `rvf` feature.
- `../ruvector-robotics` uses `rvf-runtime`/`rvf-types` under its `rvf` feature.
- `../ruvector-rulake` uses `sha3` pinned to the version used by `rvf-crypto`.
