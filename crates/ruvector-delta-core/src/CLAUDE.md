# ruvector-delta-core/src

Source for the core delta abstractions. `no_std`-capable via the `std` feature.

- `lib.rs` — module roots and crate-level docs / examples.
- `delta.rs` — `Delta` trait + `VectorDelta` (compute/apply diffs between vectors).
- `stream.rs` — `DeltaStream` ordered delta sequence (event sourcing).
- `window.rs` — `DeltaWindow` time-bounded aggregation.
- `encoding.rs` — sparse / dense delta encodings (bincode-backed).
- `compression.rs` — delta-specific compression behind the `compression` feature
  (lz4_flex, zstd).
- `error.rs` — error / result types.
