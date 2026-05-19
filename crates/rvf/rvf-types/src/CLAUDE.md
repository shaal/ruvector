# rvf-types/src

Source — one module per type group. `no_std` baseline; opt into `alloc`, `std`, `serde`, `ed25519`.

## Files

- `lib.rs` — `no_std` shim, module decls, re-exports.
- `constants.rs` — magic numbers, sizes, format version.
- `segment.rs` / `segment_type.rs` — segment header + `SegmentType` enum.
- `data_type.rs` — vector element types (f32/f16/u8/binary).
- `quant_type.rs` — quantization tier enums.
- `compression.rs` — compression-codec enums.
- `checksum.rs`, `sha256.rs` — checksum primitives.
- `signature.rs`, `ed25519.rs` — signature records and Ed25519 keys (feature-gated).
- `attestation.rs` — TEE attestation record types.
- `lineage.rs` — lineage record types.
- `witness.rs` — WITNESS_SEG record types.
- `manifest.rs` — manifest tag types.
- `delta.rs` — delta-encoded record types.
- `filter.rs` — `FilterExpr`-style types.
- `flags.rs` — segment / runtime flag bits.
- `error.rs` — `RvfError` (crate-wide error).
- `refcount.rs`, `cow_map.rs` — RVCOW data structures.
- `agi_container.rs`, `dashboard.rs`, `quality.rs` — AGI Container types (ADR-036).
- `kernel.rs`, `kernel_binding.rs` — KERNEL_SEG layout, `KernelArch`.
- `ebpf.rs` — EBPF_SEG types and `EBPF_MAGIC`.
- `qr_seed.rs` — QR Cognitive Seed payload structures.
- `wasm_bootstrap.rs` — WASM bootstrap path types.
- `membership.rs`, `profile.rs`, `security.rs` — auxiliary surfaces.
