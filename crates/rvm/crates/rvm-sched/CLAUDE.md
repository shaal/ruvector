# rvm-sched

Two-signal coherence-aware scheduler for the RVM microhypervisor (ADR-132 DC-4).

```
priority = deadline_urgency + cut_pressure_boost
```

Novelty scoring and structural risk are deferred to post-v1.

## Scheduling modes

- **Reflex** — hard real-time. Bounded local execution only; no cross-partition traffic.
- **Flow** — normal execution with coherence-aware placement.
- **Recovery** — stabilisation: replay, rollback, split.

## Constraints

Partition switch is the **hot path** — no allocation, no graph work, no policy. Switches are **not** individually witnessed (DC-10); epoch summaries are emitted instead. Coherence engine is optional (DC-1/DC-6); degraded mode uses deadline only.

`#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`.

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `rvm-partition`, `rvm-witness`, `spin`.
- `src/lib.rs` — crate root.
- `src/scheduler.rs` — main scheduling loop and priority computation.
- `src/modes.rs` — Reflex / Flow / Recovery mode state machine.
- `src/priority.rs` — priority formula (`deadline_urgency + cut_pressure_boost`).
- `src/switch.rs` — hot-path partition switch.
- `src/per_cpu.rs` — per-CPU scheduler state.
- `src/smp.rs` — SMP coordination across CPUs.
- `src/epoch.rs` — epoch tracking + summary-witness emission (DC-10).
- `src/degraded.rs` — fallback mode when coherence engine is unavailable.

See `../CLAUDE.md`.
