# rvagent-subagents/tests

- `integration_tests.rs` — end-to-end spec -> compile -> spawn -> result flow.
- `orchestrator_tests.rs` — parallel spawning, error handling, cancellation.
- `security_validation.rs` — `SubAgentResultValidator` (ADR-103 C8): length
  bounds, content checks.
