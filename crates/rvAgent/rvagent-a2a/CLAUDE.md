# rvagent-a2a

Google Agent2Agent (A2A) peer-to-peer protocol server/client for rvAgent. Implements
JSON-RPC 2.0 over HTTP, `/.well-known/agent.json` discovery, `text/event-stream`
streaming, HMAC-signed push webhooks, plus the RuVector r2/r3 extensions in
ADR-159: signed content-addressed `AgentCard`s, per-task `TaskPolicy`, pluggable
peer selectors, typed `ArtifactKind` (including `RuLakeWitness` for zero-copy
vector handoff), global budget, trace causality, and a recursion guard.

## Layout

- `Cargo.toml` — library only. Used by `rvagent-cli` to mount `server::A2aServer`.
- `src/lib.rs` — module roots and crate-level docs.
- `src/types.rs` — A2A spec types (Task, Message, Agent, etc.).
- `src/error.rs` — error taxonomy.
- `src/client.rs` — HTTP/JSON-RPC client for outbound peer calls.
- `src/server/` — axum server (see CLAUDE.md): json_rpc, push, sse handlers.
- `src/identity.rs` — signed `AgentCard`s + content-addressed `AgentID`s (r2).
- `src/policy.rs` — `TaskPolicy` enforcement (r2).
- `src/routing.rs` — pluggable `PeerSelector` (r2).
- `src/artifact_types.rs` — typed `ArtifactKind` enum incl. `RuLakeWitness` (r2).
- `src/budget.rs` — `GlobalBudget` for the dispatch queue (r3).
- `src/context.rs` — `TaskContext` trace propagation (r3).
- `src/recursion_guard.rs` — cycle / depth guard before dispatch (r3).
- `src/executor.rs` — task executor (`executor_remote` tested).
- `src/config.rs` — server configuration loader.
- `benches/` — `budget_ledger.rs`, `task_context.rs`.
- `tests/` — many integration tests covering each protocol feature.

## Related

- ADR-159 (`docs/adr/ADR-159-rvagent-a2a-protocol.md`).
- `crates/rvAgent/rvagent-acp` — typically mounted alongside this in the same
  axum binary.
