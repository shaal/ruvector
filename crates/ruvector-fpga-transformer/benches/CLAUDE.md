# ruvector-fpga-transformer/benches

Criterion benchmarks (and acts as the de-facto test suite for this crate).

## Files

- `correctness.rs` — output correctness across backends (native_sim is the reference).
- `gating.rs` — coherence + policy gate decision throughput.
- `latency.rs` — end-to-end inference latency with bounded-timing assertions.

Run via `cargo bench -p ruvector-fpga-transformer`.
