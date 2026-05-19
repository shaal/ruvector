# ruvector-temporal-tensor/src

Source for the tiered temporal tensor compressor.

- `lib.rs` — module roots, pipeline docs, ratio table.
- `compressor.rs` — `TemporalTensorCompressor`: `push_frame`, `flush`,
  `set_access`.
- `tier_policy.rs` — `TierPolicy` mapping access counts to bit widths.
- `tiering.rs` — promotion/demotion across tiers.
- `quantizer.rs` — groupwise symmetric quantize/dequantize.
- `bitpack.rs` — 3/5/7/8-bit packing.
- `delta.rs` — temporal delta encoding between consecutive segments.
- `segment.rs` — segment record layout and boundary detection.
- `coherence.rs` — coherence checks across segments.
- `store.rs` + `store_ffi.rs` — in-memory store + its FFI surface.
- `persistence.rs` (feature `persistence`) — disk-backed BlockIO + MetaLog.
- `agentdb.rs` — AgentDB store adapter.
- `f16.rs` — f16 helpers (no external crate, keeps zero-dep).
- `ffi.rs` (feature `ffi`) — C / WASM FFI exports.
- `metrics.rs` — counters / sizing telemetry.
- `core_trait.rs` — shared compressor trait.
