# ruvector-raft/fuzz/fuzz_targets

- `fuzz_raft_messages.rs` — fuzzes deserialization / handling of Raft RPC messages (`AppendEntries`, `RequestVote`, `InstallSnapshot`) to catch panics, deadlocks, or invalid-state transitions.
