# ruvix-shell/src/commands

Built-in shell command implementations. Each command file exposes a function dispatched by `mod.rs` based on the parsed command
name.

## Files

- `mod.rs` — command registry / dispatch.
- `info.rs` — `info`: kernel version, boot time, uptime.
- `mem.rs` — `mem`: memory statistics.
- `tasks.rs` — `tasks`: task listing.
- `caps.rs` — `caps`: capability table dump.
- `queues.rs` — `queues`: queue statistics.
- `vectors.rs` — `vectors`: vector-store info.
- `proofs.rs` — `proofs`: proof-subsystem statistics.
- `cpu.rs` — `cpu`: per-CPU info (SMP).
- `witness.rs` — `witness`: witness-log viewer.
- `perf.rs` — `perf`: performance counters.
