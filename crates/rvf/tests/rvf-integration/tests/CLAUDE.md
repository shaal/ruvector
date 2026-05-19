# rvf-integration-tests/tests

Integration / acceptance tests for the RVF crate family. One Rust file per concern.

## Files (grouped)

### End-to-end lifecycle
- `e2e_store_lifecycle.rs` — full open/insert/query/delete/compact loop.
- `e2e_multi_segment.rs` — multi-segment scenarios.
- `e2e_progressive_recall.rs` — Layer A/B/C progressive recall.
- `e2e_quantization_tiers.rs` — hot/warm/cold tier behaviour.
- `e2e_wire_interop.rs` — interop across wire-format versions.
- `e2e_crash_safety.rs` — crash recovery.
- `runtime_lifecycle.rs` — runtime-level lifecycle.
- `rvf_smoke_test.rs` — basic smoke.

### Wire / segment
- `wire_round_trip.rs` — round-trip every segment type.
- `bit_flip_detection.rs` — checksum / signature detect bit-flips.
- `segment_preservation.rs`, `unknown_segment_preservation.rs` — forward-compat across reader versions.
- `extension_aliasing.rs` — extension/aliasing rules.
- `cross_platform_compat.rs` — endian / arch compatibility.
- `file_identity.rs` — UUID / fingerprint stability.

### Manifest / index
- `manifest_boot.rs` — boot from L0/L1.
- `index_recall.rs` — HNSW recall acceptance gate.
- `filter_traversal.rs` — filter expressions over the index.

### Crypto / lineage / attestation
- `crypto_sign_verify.rs` — Ed25519 sign/verify.
- `attestation_witness.rs` — TEE attestation + WITNESS_SEG.
- `lineage_derivation.rs`, `lineage_verification.rs` — derive child stores; verify lineage.

### Quantization
- `quant_accuracy.rs` — accuracy targets for scalar/PQ/binary.

### Cognitive container / RVCOW
- `computational_container.rs` — cognitive-container build/launch path.
- `kernel_selection.rs` — kernel arch selection.
- `cow_branching.rs`, `cow_crash_recovery.rs`, `cow_benchmarks.rs` — RVCOW (ADR-031) coverage.

### Profile / CLI
- `profile_compat.rs` — Core/RVText profile compatibility.
- `rvf_cli_smoke.rs` — smoke-tests the `rvf` binary from `../../../rvf-cli`.
