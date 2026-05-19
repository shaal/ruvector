# ruvector-raft/src

Raft consensus source.

## Files

- `lib.rs` — public re-exports and `RaftError` enum.
- `node.rs` — `RaftNode`, `RaftNodeConfig`: drives the state machine; tokio-based.
- `state.rs` — `RaftState`, `PersistentState` (currentTerm, votedFor, log), `VolatileState` (commitIndex, lastApplied), `LeaderState` (nextIndex, matchIndex).
- `election.rs` — leader election, randomized election timer.
- `log.rs` — append-only log: append, truncate, persist (via `bincode`).
- `rpc.rs` — RPC request/response types: `AppendEntriesRequest/Response`, `RequestVoteRequest/Response`, `InstallSnapshotRequest/Response`.
