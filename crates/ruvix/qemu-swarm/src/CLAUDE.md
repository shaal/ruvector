# ruvix-qemu-swarm/src

Source for the QEMU-based distributed RuVix simulator.

## Files

- `lib.rs` — crate root + re-exports.
- `main.rs` — binary entry point: parses a TOML cluster config and drives the orchestrator.
- `config.rs` — TOML cluster config types (`Cluster`, `Node`, ...).
- `cluster.rs` — `Cluster` runtime representation.
- `node.rs` — per-`Node` lifecycle (spawn QEMU child, manage stdio).
- `orchestrator.rs` — top-level orchestrator coordinating all nodes.
- `consensus.rs` — consensus / coordination helpers used by demos.
- `network.rs` — virtual networking glue between QEMU instances.
- `console.rs` — serial console multiplexer.
- `monitor.rs` — health and status monitoring.
- `error.rs` — crate error enum.
