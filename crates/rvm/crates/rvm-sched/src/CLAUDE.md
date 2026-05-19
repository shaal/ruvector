# rvm-sched/src

- `lib.rs` — crate root.
- `scheduler.rs` — main scheduling loop, selects next partition by priority.
- `modes.rs` — Reflex / Flow / Recovery state machine.
- `priority.rs` — `priority = deadline_urgency + cut_pressure_boost`.
- `switch.rs` — partition-switch hot path (no allocation, no graph work).
- `per_cpu.rs` — per-CPU scheduler state.
- `smp.rs` — SMP coordination across CPUs.
- `epoch.rs` — epoch tracking and summary witness emission (DC-10).
- `degraded.rs` — degraded fallback when coherence engine unavailable.

See `../CLAUDE.md`.
