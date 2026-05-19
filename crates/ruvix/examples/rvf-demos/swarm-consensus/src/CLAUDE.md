# rvf-swarm-demo/src

## Files

- `main.rs` — binary entry: spawns N agent tasks via `ruvix-nucleus`, wires them with capability-protected queues, and drives a
  consensus round using the coherence-aware scheduler.
