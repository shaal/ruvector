# ruvix-sched/src

## Files

- `lib.rs` — crate root; re-exports `Scheduler`, `SchedulerConfig`, `TaskControlBlock`, `TaskState`.
- `scheduler.rs` — `Scheduler` orchestration with EDF + novelty + risk signals.
- `task.rs` — `TaskControlBlock` + `TaskState` (Ready / Running / Blocked / etc.).
- `priority.rs` — priority arithmetic combining the three signals.
- `partition.rs` — per-RVF-mount partition scheduling.
- `novelty.rs` — novelty signal: vector-distance from recent inputs (optionally `ruvector-coherence`).
- `error.rs` — scheduler error enum.
