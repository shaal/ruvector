# rvf-adapter-sona/src

Source.

## Files

- `lib.rs` — public re-exports + ADR-029 architecture docs.
- `config.rs` — `SonaConfig` (data dir, dim, replay capacity, trajectory window).
- `trajectory.rs` — `TrajectoryStore` recording sequences of state embeddings.
- `experience.rs` — `ExperienceReplayBuffer` (circular `(s, a, r, s')` buffer for off-policy training).
- `pattern.rs` — `NeuralPatternStore` (recognised patterns with confidence; searchable by category or embedding similarity).
