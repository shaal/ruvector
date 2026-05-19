# rvagent-core/tests

Unit / integration tests for the core types.

- `config_tests.rs` — `RvAgentConfig` parsing, defaults, security policy.
- `state_tests.rs` — `AgentState`, `FileData`, todos, Arc semantics.
- `message_tests.rs` — `Message`/`ToolCall` round-trips.
- `model_tests.rs` — `ChatModel` / model resolution.
- `integration_tests.rs` — cross-module integration (graph + state + messages).
