# ruvllm/src/reasoning_bank

Production-grade learning from Claude (and other LLM) trajectories. Records
execution paths with quality metrics, indexes patterns in HNSW (150x faster
similarity search), analyzes verdicts for failure root cause, runs EWC++
consolidation to avoid catastrophic forgetting, and distills old trajectories
to compress while preserving lessons.

State persists in `../../.reasoning_bank_patterns` (crate root, binary).

## Files
- `mod.rs` - public API + architecture diagram.
- `trajectory.rs` - real-time trajectory recorder.
- `pattern_store.rs` - HNSW-indexed pattern storage.
- `verdicts.rs` - enhanced verdict system (success/failure analysis,
  root-cause detection).
- `consolidation.rs` - EWC++ consolidation step.
- `distillation.rs` - memory distillation (compress old trajectories).
