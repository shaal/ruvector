# rvagent-subagents/src

Source for subagent spec/compile/orchestrate/validate.

- `lib.rs` — module roots + re-exports for the result validator, CRDT merge,
  and orchestrator types.
- `builder.rs` — `SubAgentSpec` -> `CompiledSubAgent` build path.
- `orchestrator.rs` — `SubAgentOrchestrator`, `spawn_parallel`, `SpawnError`.
- `crdt_merge.rs` — `CrdtState`, `VectorClock`, `merge_subagent_results`,
  `MergeError`. See `../CRDT_MERGE.md`.
- `result_validator.rs` — security validator (C8): `SubAgentResultValidator`,
  `ValidationConfig`, `ValidationError`, `DEFAULT_MAX_RESPONSE_LENGTH`.
- `validator.rs` — additional spec-time validation.
- `prompts.rs` — subagent prompt templates.
