# ruvix-queue

io_uring-style ring buffer IPC for the RuVix Cognition Kernel (ADR-087 Section 7). All inter-task communication goes through
queues — there are no synchronous IPC calls, no shared memory without explicit region grants, no signals. Uses separate
submission (SQ) and completion (CQ) rings with atomic head/tail for lock-free operation.

Zero-copy semantics apply when sender and receiver share a region: only Immutable / AppendOnly regions may use descriptors
(TOCTOU protection — Slab descriptors are rejected, ADR-087 Section 20.5).

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-region`. Dev: criterion, proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/queue_bench.rs` — SQ/CQ throughput.
- `tests/integration.rs` — integration tests.
