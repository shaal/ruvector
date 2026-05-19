# rvm-partition

Partition object model for the RVM microhypervisor (ADR-133). A partition is **not** a VM: no emulated hardware, no guest BIOS, no virtual device model. It is a container for a scoped capability table, communication edges, coherence / cut-pressure metrics, and CPU affinity / VMID.

Partitions are the unit of scheduling, isolation, migration, and fault containment. Every lifecycle transition emits a witness record.

## Constraints

Max 256 partitions per RVM instance (ARM VMID width); partition switch target < 10 us; scheduler uses `deadline_urgency + cut_pressure_boost`; coherence engine is optional (DC-1). `#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`.

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `rvm-cap`, `rvm-witness`, `spin`.
- `src/lib.rs` — crate root and module wiring.
- `src/partition.rs` — `Partition` struct + invariants.
- `src/manager.rs` — `PartitionManager` (lookup, allocation).
- `src/lifecycle.rs` — partition lifecycle state machine.
- `src/ops.rs` — partition-level operations (run / pause / resume).
- `src/cap_table.rs` — scoped capability table attached to a partition.
- `src/comm_edge.rs` — `CommEdge` (weighted edge in the coherence graph).
- `src/device.rs` — device lease bookkeeping for the partition.
- `src/ipc.rs` — inter-partition IPC primitives.
- `src/split.rs`, `src/merge.rs` — novel split / merge operations with strict preconditions (witness-gated).

See `../CLAUDE.md`.
