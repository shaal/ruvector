# ruvLLM / src / sona / loops

The three learning loops that drive SONA continual learning.

## Important files
- `mod.rs` - module root.
- `instant.rs` - the instant loop: applies micro-updates synchronously on each interaction.
- `background.rs` - the background loop: replays trajectories from the ReasoningBank, runs EWC++ regularisation.
- `coordinator.rs` - the coordinator loop: schedules instant + background work and resolves conflicts.

## Related
- Spec: `../../../docs/SONA/02-LEARNING-LOOPS.md`. Engine glue: `../engine.rs`. Pattern store: `../reasoning_bank.rs`.
