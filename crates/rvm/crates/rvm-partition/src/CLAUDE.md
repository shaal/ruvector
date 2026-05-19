# rvm-partition/src

- `lib.rs` — crate root.
- `partition.rs` — `Partition` struct: scoped capability table, comm edges, coherence + cut-pressure metrics, CPU affinity, VMID.
- `manager.rs` — `PartitionManager` (allocation, lookup, max-256 enforcement).
- `lifecycle.rs` — lifecycle state machine; every transition emits a witness.
- `ops.rs` — high-level partition operations (run / pause / resume / migrate).
- `cap_table.rs` — scoped capability table attached to a partition.
- `comm_edge.rs` — `CommEdge` (weighted edge in the coherence graph).
- `device.rs` — device-lease bookkeeping.
- `ipc.rs` — inter-partition IPC primitives.
- `split.rs` — partition split (witness-gated, strict preconditions).
- `merge.rs` — partition merge (witness-gated, strict preconditions).

See `../CLAUDE.md`.
