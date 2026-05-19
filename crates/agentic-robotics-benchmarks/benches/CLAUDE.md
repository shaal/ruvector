# agentic-robotics-benchmarks/benches

Criterion benchmark sources for the agentic robotics runtime. Each file is a standalone `harness = false` bench target declared in
the parent `Cargo.toml`.

## Files

- `message_serialization.rs` — serde/`serde_json` round-trip cost for `agentic-robotics-core` message types.
- `pubsub_latency.rs` — latency of the in-process pub/sub bus from `agentic-robotics-core`.
- `executor_performance.rs` — throughput and scheduling overhead of the `agentic-robotics-rt` tokio-based executor.

Run a specific bench: `cargo bench -p agentic-robotics-benchmarks --bench pubsub_latency`.
