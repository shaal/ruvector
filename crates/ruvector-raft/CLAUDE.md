# ruvector-raft

Production Raft consensus implementation for distributed metadata in the RuVector vector database. Follows the Raft paper specification (leader election, log replication, snapshots, membership changes).

## Important files

- `Cargo.toml` — deps: `ruvector-core`, `tokio` (with `time`), `serde`, `bincode`, `dashmap`, `parking_lot`, `chrono`, `uuid`, `futures`, `rand`.
- `src/lib.rs` — public re-exports (`RaftNode`, `RaftNodeConfig`, RPC request/response types, `LeaderState`, `PersistentState`, `RaftState`, `VolatileState`) and `RaftError` enum.

## Module map (src/)

- `node.rs` — `RaftNode`, `RaftNodeConfig`: top-level node state machine.
- `state.rs` — `RaftState`, `PersistentState`, `VolatileState`, `LeaderState` — Raft state machine partitioning.
- `election.rs` — leader election + election timer.
- `log.rs` — append-only log with truncation and persistence.
- `rpc.rs` — `AppendEntries`, `RequestVote`, `InstallSnapshot` request/response types.

## Tests & fuzzing

- `tests/integration_tests.rs` — integration tests for the node state machine.
- `fuzz/` — `cargo-fuzz` harness; `fuzz_targets/fuzz_raft_messages.rs` fuzzes RPC message parsing.

## Public API surface

`RaftNode`, `RaftNodeConfig`, `RaftError`, `RaftResult<T>`, RPC types, state types.

## Related

- `crates/ruvector-core` — backing storage for log/snapshot persistence.
- `crates/ruvector-snapshot` — snapshot machinery used by `InstallSnapshot`.
