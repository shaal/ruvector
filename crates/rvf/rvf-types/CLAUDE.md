# rvf-types

Core types for the RuVector Format (RVF) — segment headers, type enums, flags, error codes, format constants. All types are `no_std` compatible by default; `std` and `alloc` are opt-in features. Foundational dependency for every other `rvf-*` crate.

## Layout

- `Cargo.toml` — features: `default = []`, `alloc`, `std = ["alloc"]`, `serde`, `ed25519`. Optional deps: `serde`, `ed25519-dalek`, `rand_core`.
- `src/lib.rs` — `no_std` shim, module decls.

## Modules

- `constants.rs` — format constants (magic numbers, sizes, version).
- `segment.rs` / `segment_type.rs` — segment header layout + `SegmentType` enum.
- `data_type.rs` — vector element types.
- `quant_type.rs` — quantization tier enums.
- `compression.rs` — compression codec enums.
- `checksum.rs`, `sha256.rs` — checksum primitives.
- `signature.rs`, `ed25519.rs` — signature record + Ed25519 key types (gated).
- `attestation.rs` — TEE attestation record types.
- `lineage.rs` — lineage / derivation record types.
- `witness.rs` — WITNESS_SEG record types.
- `manifest.rs` — manifest type tags.
- `delta.rs` — delta-encoded record types.
- `filter.rs` — filter expression types.
- `flags.rs`, `error.rs` — flags + `RvfError`.
- `refcount.rs`, `cow_map.rs` — RVCOW data structures.
- `agi_container.rs`, `dashboard.rs`, `quality.rs` — AGI container types.
- `kernel.rs`, `kernel_binding.rs` — KERNEL_SEG layout (`KernelArch`, etc.).
- `ebpf.rs` — `EbpfHeader`, `EbpfProgramType`, `EbpfAttachType`, `EBPF_MAGIC`.
- `qr_seed.rs` — QR Cognitive Seed payload types.
- `wasm_bootstrap.rs` — types for the WASM bootstrap path.
- `membership.rs`, `profile.rs`, `security.rs` — auxiliary surfaces.

## Public API

`RvfError`, every segment-related struct/enum re-exported from `lib.rs`. Look here when reading the binary format spec.

## Related

- Every `rvf-*` crate depends on this
- `../../ruvector-domain-expansion`, `../../ruvector-robotics` re-export `rvf-types` under their `rvf` feature
