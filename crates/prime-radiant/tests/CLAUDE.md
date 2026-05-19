# prime-radiant/tests

Integration + property + scenario tests.

## Files

- `chaos_tests.rs` — chaos / fault injection tests.
- `gpu_coherence_tests.rs` — wgpu pipeline correctness vs CPU baseline.
- `replay_determinism.rs` — replays produce identical witnesses (auditability).
- `ruvllm_integration_tests.rs` — end-to-end gate of LLM token streams.
- `storage_tests.rs` — memory/file/postgres backend round trips.

## Subdirectories

- `integration/` — coherence/gate/governance/graph integration tests.
- `property/` — proptest invariants.
