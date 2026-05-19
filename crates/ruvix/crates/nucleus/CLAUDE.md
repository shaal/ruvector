# ruvix-nucleus

Integration crate for the RuVix Cognition Kernel. Brings all RuVix subsystems together and provides: the syscall dispatch table
for all 12 syscalls defined in ADR-087, the top-level `Kernel` struct coordinating all subsystems, deterministic replay support
for checkpoint/restore, and the witness log for attestation and auditability.

## Architecture

```
+-----------------------+
|     RuVix Nucleus     |
+-----------+-----------+
            |
   +--------+--------+--------+--------+
   |        |        |        |        |
RegionMgr CapMgr  QueueMgr ProofEngine ...
   |        |        |        |
VectorMgr GraphMgr Scheduler WitnessLog
```

## Files

- `Cargo.toml` — depends on all subsystem crates: `types`, `region`, `cap`, `queue`, and optionally `shell`.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `tests/acceptance.rs`, `tests/deterministic_replay.rs`, `tests/syscall_tests.rs` — top-level acceptance + replay + per-syscall
  tests.
- `benches/syscall_bench.rs` — end-to-end syscall dispatch latency.
