# agentic-robotics-rt/src

Source for the ROS3 real-time runtime.

- `lib.rs` — module roots and the public `RTPriority` enum (Background, Low, Normal, High,
  Critical with `u8` conversions).
- `executor.rs` — `ROS3Executor`, `Priority`, `Deadline`. Wraps tokio for soft-RT plus a
  priority-aware queue for hard-RT-style tasks.
- `scheduler.rs` — `PriorityScheduler`, the priority queue dispatcher.
- `latency.rs` — `LatencyTracker` built on `hdrhistogram` for latency percentiles.

All public surface is re-exported via `lib.rs`.
