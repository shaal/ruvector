# ruvector-temporal-tensor

Temporal tensor compression with tiered quantization (ADR-017). Groupwise
symmetric quantization with temporal segment reuse and access-pattern-driven
tier selection (8/7/5/3 bit). Pipeline: `f32 frame -> tier_policy -> quantizer ->
bitpack -> segment` and the inverse on read. Zero external dependencies on the
default build (WASM-friendly).

Compression ratios vs f32: Hot 8-bit ~4.0x, Warm 7-bit ~4.57x, Warm 5-bit ~6.4x,
Cold 3-bit ~10.67x.

## Layout

- `Cargo.toml` — `crate-type = ["lib"]`. Features:
  - `ffi` — enable WASM / C FFI exports.
  - `simd` — SIMD-accelerated quantization (future).
  - `persistence` — disk-backed `BlockIO` + `MetaLog` (uses `std::fs`).
- `src/lib.rs` — module roots + pipeline / ratio docs.
- `src/compressor.rs` — `TemporalTensorCompressor` public API.
- `src/tier_policy.rs` — `TierPolicy` (access-pattern -> tier mapping).
- `src/tiering.rs` — tier transitions / migration.
- `src/quantizer.rs` — groupwise symmetric quantize/dequantize.
- `src/bitpack.rs` — 3/5/7/8-bit bit packing.
- `src/delta.rs` — temporal delta encoding.
- `src/segment.rs` — segment layout / boundaries.
- `src/coherence.rs` — segment coherence checks.
- `src/store.rs`, `src/store_ffi.rs` — in-memory store + FFI surface.
- `src/persistence.rs` — disk-backed store (feature `persistence`).
- `src/agentdb.rs` — AgentDB integration.
- `src/f16.rs` — f16 helpers (no external crate).
- `src/ffi.rs` — C/WASM FFI exports (feature `ffi`).
- `src/metrics.rs`, `src/core_trait.rs` — supporting types.
- `tests/` — integration, persistence, property, stress, benchmarks, WASM FFI.

## Public API

`TemporalTensorCompressor`, `TierPolicy`, plus segment / store types.

## Related

- AgentDB integration (`src/agentdb.rs`); other vector-storage crates.
