# ruvLLM / src / sona

SONA - Self-Organizing Neural Adapters - the continual-learning subsystem of ruvLLM. Combines LoRA Ultra adapters, EWC++ catastrophic-forgetting resistance, a ReasoningBank pattern store, trajectory tracking, and three learning loops (instant / background / coordinator).

## Important files
- `mod.rs` - module root.
- `engine.rs` - SONA engine that owns the loops and adapter pool.
- `lora.rs` - LoRA Ultra adapter implementation.
- `ewc.rs` - Elastic Weight Consolidation++.
- `reasoning_bank.rs` - persistent pattern store (companion to the `reasoningbank-*` skills).
- `trajectory.rs` - trajectory tracking and replay.
- `types.rs` - shared SONA types.
- `loops/` - the learning loops themselves.

## Related
- Spec: `../../docs/SONA/`. Benches: `../../benches/sona_bench.rs`. Integration tests: `../../tests/sona_integration.rs`. Workspace crate: `../../../crates/sona/`.
