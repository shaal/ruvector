# agentic-robotics-rt/benches

Criterion benchmarks for the real-time executor.

- `latency.rs` — registered as `[[bench]] name = "latency"` in Cargo.toml. Measures task
  dispatch / deadline latency through the `ROS3Executor` and `PriorityScheduler`.

Run: `cargo bench -p agentic-robotics-rt --bench latency`.
