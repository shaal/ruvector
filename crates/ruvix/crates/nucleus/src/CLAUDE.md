# ruvix-nucleus/src

## Files

- `lib.rs` — crate root, top-level docs and re-exports.
- `kernel.rs` — `Kernel` struct: owns all subsystem managers and exposes the public entry surface.
- `syscall.rs` — syscall dispatch table for all 12 ADR-087 syscalls.
- `vector_store.rs` — wiring between `Kernel` and `ruvix-vecgraph::KernelVectorStore`.
- `graph_store.rs` — wiring for the graph store.
- `proof_engine.rs` — integration of `ruvix-proof`.
- `scheduler.rs` — integration of `ruvix-sched`.
- `checkpoint.rs` — checkpoint/restore for deterministic replay.
- `witness_log.rs` — attestation/audit log appended on every successful mutation.
- `shell_backend.rs` — optional `ShellBackend` implementation that connects `ruvix-shell` commands to live kernel state.
