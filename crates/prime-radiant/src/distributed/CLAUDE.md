# prime-radiant/src/distributed

Distributed coherence state sharing — adapter into `ruvector-raft` for multi-node deployments.

## Files

- `mod.rs` — module entry.
- `config.rs` — distributed config (node id, cluster topology, replication).
- `adapter.rs` — Raft adapter wiring `CoherenceEngine` state into a replicated log.
- `state.rs` — replicated state machine type for coherence updates.

## Related

- `crates/ruvector-raft` — underlying Raft implementation (optional dep).
