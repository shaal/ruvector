# rvagent-core

Core types for the rvAgent framework. Provides the typed `AgentState` (Arc-based
O(1) cloning), `AgentGraph` execution graph, `RvAgentConfig` (security/budget),
model resolution, message types, prompt builder, copy-on-write state backend,
bump arena, lock-free metrics, parallel execution helpers, AGI containers
(RVF-segment based), session crypto, and RVF bridge.

Implements ADR-103 review amendments (A2 parallel, A8 arena, A9 metrics, B1 AGI
containers, B2 CoW state, B4 budget enforcement).

## Layout

- `Cargo.toml` — lib. Deps: tokio, serde, dashmap, parking_lot, async-trait,
  smallvec, aes-gcm, sha3, rand, uuid, chrono, anyhow. Two benches.
- `src/lib.rs` — module roots + re-exports.
- `src/agi_container.rs` — `AgiContainerBuilder`, `ParsedContainer`,
  `SegmentType`, `SkillDefinition`, `ToolDefinition`, `OrchestratorConfig`,
  `AgentPrompt`, `AgentNode as AgiAgentNode`, `ContainerError`, `agi_tags` (B1).
- `src/arena.rs` — bump arena for hot-path scratch allocations (A8).
- `src/budget.rs` — `BudgetEnforcer`, `BudgetError`, `BudgetUtilization` (B4).
- `src/config.rs` — `RvAgentConfig`, `BackendConfig`, `ResourceBudget`,
  `SecurityPolicy`.
- `src/cow_state.rs` — `CowStateBackend` (B2). See `docs/cow_state.md`.
- `src/error.rs` — `RvAgentError`, `Result`.
- `src/graph.rs` — `AgentGraph`, `AgentNode`, `GraphConfig`, `ToolExecutor`.
- `src/messages.rs` — `Message`, `AiMessage`, `HumanMessage`, `SystemMessage`,
  `ToolMessage`, `ToolCall`.
- `src/metrics.rs` — lock-free metrics (A9).
- `src/models.rs` — `ChatModel`, `StreamingChatModel`, `ModelConfig`, `Provider`,
  `StreamChunk`, `StreamUsage`.
- `src/parallel.rs` — parallel async execution utilities (A2).
- `src/prompt.rs` — `SystemPromptBuilder`, `BASE_AGENT_PROMPT`.
- `src/rvf_bridge.rs` — RVF manifest/witness bridge (`RvfManifest`,
  `RvfManifestEntry`, `RvfMountHandle`, `RvfWitnessHeader`, `PolicyCheck`,
  `GovernanceMode`, etc.).
- `src/session_crypto.rs` — `SessionCrypto`, `EncryptionKey`, `generate_key`,
  `derive_key`, `generate_session_filename` (AES-GCM + SHA3).
- `src/state.rs` — `AgentState`, `FileData`, `SkillMetadata`, `TodoItem`,
  `TodoStatus`.
- `src/string_pool.rs` — thread-safe string interning.
- `benches/` — `state_bench.rs`, `rvf_bridge_bench.rs`.
- `docs/cow_state.md` — CoW state backend design notes.
- `examples/` — `agi_container_demo.rs`, `cow_state_demo.rs`,
  `session_crypto_demo.rs`.
- `tests/` — `config_tests.rs`, `integration_tests.rs`, `message_tests.rs`,
  `model_tests.rs`, `state_tests.rs`.
- `C9_IMPLEMENTATION_SUMMARY.md`, `IMPLEMENTATION_C9.md` — implementation notes.

## Related

Used by every other `rvAgent/*` crate.
