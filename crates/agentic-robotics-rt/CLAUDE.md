# agentic-robotics-rt

ROS3 real-time execution layer. Combines Tokio (soft real-time async) with a priority-aware
scheduler to drive robotics workloads on top of `agentic-robotics-core`. Provides task priority
levels, deadline tracking, and HDR-histogram latency telemetry.

## Layout

- `Cargo.toml` — depends on `agentic-robotics-core`, tokio, crossbeam, rayon, hdrhistogram.
  Defines `[[bench]] latency`.
- `src/lib.rs` — module roots; declares `RTPriority` (Background..Critical) and re-exports
  `ROS3Executor`, `Priority`, `Deadline`, `PriorityScheduler`, `LatencyTracker`.
- `src/executor.rs` — `ROS3Executor` and priority/deadline types.
- `src/scheduler.rs` — `PriorityScheduler` for priority-aware task dispatch.
- `src/latency.rs` — `LatencyTracker` using `hdrhistogram`.
- `benches/latency.rs` — Criterion latency benchmark.

## Public API

`RTPriority`, `ROS3Executor`, `Priority`, `Deadline`, `PriorityScheduler`, `LatencyTracker`.

## Related

- `crates/agentic-robotics-core` — primitives consumed here.
