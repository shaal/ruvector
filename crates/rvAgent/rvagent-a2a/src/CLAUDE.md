# rvagent-a2a/src

Source for the A2A protocol implementation.

- `lib.rs` — module roots and crate docs (spec + r2/r3 extensions).
- `types.rs` — A2A spec types (Task, Message, AgentCard, etc.).
- `error.rs` — error taxonomy.
- `client.rs` — outbound JSON-RPC client.
- `server/` — axum-based server (see its own CLAUDE.md).
- `identity.rs` (r2) — signed `AgentCard`s + content-addressed `AgentID`s.
- `policy.rs` (r2) — per-task `TaskPolicy` enforcement.
- `routing.rs` (r2) — pluggable `PeerSelector`.
- `artifact_types.rs` (r2) — typed `ArtifactKind` incl. `RuLakeWitness`.
- `budget.rs` (r3) — `GlobalBudget` for the dispatch queue.
- `context.rs` (r3) — `TaskContext` trace propagation.
- `recursion_guard.rs` (r3) — cycle / depth guard.
- `executor.rs` — task executor.
- `config.rs` — server config loader.
