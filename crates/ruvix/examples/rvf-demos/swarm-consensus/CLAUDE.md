# rvf-swarm-demo (swarm-consensus)

RVF Swarm Demo — multi-agent coordination running on RuVix. Demonstrates how several agent tasks negotiate consensus using
capabilities + queues + the coherence-aware scheduler.

## Files

- `Cargo.toml` — `publish = false`. Depends on `ruvix-nucleus`, `ruvix-types`, `ruvix-cap`, `ruvix-queue`, `ruvix-sched`.
  Feature: `default = ["std"]`, `std`.
- `swarm.rvf.json` — RVF manifest describing the multi-agent package.
- `src/main.rs` — binary entry point that boots the swarm and drives the consensus protocol.
