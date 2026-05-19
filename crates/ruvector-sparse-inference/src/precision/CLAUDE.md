# ruvector-sparse-inference/src/precision

Mixed-precision (3/5/7-bit) quantization subsystem.

- `mod.rs` — module roots and public re-exports.
- `lanes.rs` — `PrecisionLane` enum (Bit3 / Bit5 / Bit7) and lane utilities.
- `policy.rs` — graduation policies that promote/demote tensors between lanes
  based on usage and drift signals.
- `quantizers.rs` — per-lane quantize / dequantize kernels.
- `telemetry.rs` — counters / drift telemetry feeding the policy.

Covered by `tests/unit/quantization_tests.rs`.
