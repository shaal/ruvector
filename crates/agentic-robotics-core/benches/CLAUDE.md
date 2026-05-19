# agentic-robotics-core/benches

Criterion benchmarks for the ROS3 messaging fabric.

## Files

- `message_passing.rs` — End-to-end pub/sub round-trip latency, recorded via `hdrhistogram`. Targets microsecond-scale determinism per crate-level goal. Configured as `harness = false` in `Cargo.toml`.

## Run

```
cargo bench -p agentic-robotics-core
```
