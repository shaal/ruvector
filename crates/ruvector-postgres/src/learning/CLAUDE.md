# ruvector-postgres/src/learning

Self-Learning and ReasoningBank module — adaptive query optimization using trajectory tracking, pattern extraction, and learned parameter optimization.

## Files

- `mod.rs` — Re-exports `OptimizationTarget`, `SearchOptimizer`, `SearchParams`, `LearnedPattern`, `PatternExtractor`, `ReasoningBank`, `QueryTrajectory`, `TrajectoryTracker`.
- `trajectory.rs` — `QueryTrajectory`, `TrajectoryTracker` — records executed queries.
- `patterns.rs` — `LearnedPattern`, `PatternExtractor` — mines patterns from trajectories.
- `reasoning_bank.rs` — `ReasoningBank` — persistent store of distilled patterns.
- `optimizer.rs` — `SearchOptimizer`, `SearchParams`, `OptimizationTarget` — chooses parameters using the bank.
- `operators.rs` — pgrx SQL surface.

## Pointers

- Backbone: `ruvector-sona` (used via `../sona/`).
- See `../../docs/learning/IMPLEMENTATION_SUMMARY.md`, `../../docs/LEARNING_MODULE_README.md`.
