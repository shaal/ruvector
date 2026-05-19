# rvagent-subagents

Subagent specification, compilation, orchestration, and result validation
(ADR-103 C8). Provides:

- `SubAgentSpec` — declarative subagent definition.
- `CompiledSubAgent` — spec compiled into a runnable graph.
- `SubAgentResult` — outcome of a subagent execution.
- `SubAgentOrchestrator` — spawn / parallel execution.
- `SubAgentResultValidator` — security validation (response length, content).
- CRDT-based result merging via vector clocks for safe parallel merges.

## Layout

- `Cargo.toml` — lib.
- `CRDT_MERGE.md` — design notes for the CRDT result-merge subsystem.
- `src/lib.rs` — module roots + re-exports.
- `src/builder.rs` — spec / compiled subagent builder.
- `src/orchestrator.rs` — `SubAgentOrchestrator`, `spawn_parallel`, `SpawnError`.
- `src/crdt_merge.rs` — `CrdtState`, `VectorClock`, `merge_subagent_results`,
  `MergeError`.
- `src/result_validator.rs` — `SubAgentResultValidator`, `ValidationConfig`,
  `ValidationError`, `DEFAULT_MAX_RESPONSE_LENGTH`.
- `src/validator.rs` — additional spec validation.
- `src/prompts.rs` — subagent prompt templates.
- `examples/crdt_merge_demo.rs` — CRDT merge demo.
- `tests/` — `integration_tests.rs`, `orchestrator_tests.rs`,
  `security_validation.rs`.

## Related

`rvagent-core`, `rvagent-middleware/subagents.rs`.
