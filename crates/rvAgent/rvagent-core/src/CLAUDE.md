# rvagent-core/src

Source for the rvAgent core types (ADR-103).

- `lib.rs` — module roots + top-level re-exports.
- `agi_container.rs` (B1) — AGI container building from RVF segments.
- `arena.rs` (A8) — bump arena for hot-path scratch allocations.
- `budget.rs` (B4) — resource budget enforcer / errors / utilization.
- `config.rs` — `RvAgentConfig`, `BackendConfig`, `ResourceBudget`,
  `SecurityPolicy`.
- `cow_state.rs` (B2) — `CowStateBackend` (see `../docs/cow_state.md`).
- `error.rs` — `RvAgentError`, `Result`.
- `graph.rs` — `AgentGraph`, `AgentNode`, `GraphConfig`, `ToolExecutor` (typed
  agent state machine).
- `messages.rs` — `Message`, `Ai/Human/System/ToolMessage`, `ToolCall`.
- `metrics.rs` (A9) — lock-free metrics.
- `models.rs` — `ChatModel`/`StreamingChatModel` traits + config.
- `parallel.rs` (A2) — parallel async execution helpers.
- `prompt.rs` — `SystemPromptBuilder`, `BASE_AGENT_PROMPT`.
- `rvf_bridge.rs` — RVF manifest / witness / mount bridge.
- `session_crypto.rs` — AES-GCM + SHA3 session encryption helpers.
- `state.rs` — `AgentState` (Arc-cloneable), `FileData`, `SkillMetadata`,
  `TodoItem`, `TodoStatus`.
- `string_pool.rs` — thread-safe string interning.
