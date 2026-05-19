# ruvector-delta-core

Core delta types and traits for behavioral vector change tracking. Provides the
foundational abstractions for computing, applying, and composing deltas on vector
data structures: deltas, delta streams (event sourcing), delta windows (time
aggregates), sparse/dense encodings, and delta-specific compression. `no_std`-capable.

## Layout

- `Cargo.toml` — `crate-type = ["lib"]`. Features:
  - `default = ["std"]`
  - `std` — opt out for `no_std`.
  - `simd` — pulls in `simsimd`.
  - `serde` — `serde`, `serde_json`, smallvec serde.
  - `compression` — `lz4_flex`, `zstd`.
  Always-on deps: thiserror, bincode 2, parking_lot, smallvec, arrayvec.
- `src/lib.rs` — `#![cfg_attr(not(feature = "std"), no_std)]`; module roots.
- `src/delta.rs` — `Delta` trait and `VectorDelta` (compute / apply between vectors).
- `src/stream.rs` — `DeltaStream` ordered delta sequence.
- `src/window.rs` — `DeltaWindow` time-bounded aggregation.
- `src/encoding.rs` — sparse / dense delta encodings.
- `src/compression.rs` — delta-specific compression (lz4 / zstd).
- `src/error.rs` — error types.

## Public API

`Delta`, `VectorDelta`, `DeltaStream`, `DeltaWindow`, plus encoding/compression
helpers. Example: `VectorDelta::compute(&old, &new)` then `delta.apply(&mut old)`.

## Related

Used by higher-level streaming/incremental vector subsystems in the workspace.
