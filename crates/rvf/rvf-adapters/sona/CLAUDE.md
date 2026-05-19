# rvf-adapter-sona

RVF adapter for SONA (Self-Optimizing Neural Architecture) per ADR-029. Stores learning trajectories, experience-replay buffers, and recognised neural patterns in a single underlying RVF file; the three data types are distinguished by a type marker in metadata field 4.

## Layout

- `Cargo.toml` — name `rvf-adapter-sona`. Deps: `rvf-runtime`, `rvf-types` (`std`). Dev: `tempfile`.
- `src/lib.rs` — re-exports `SonaConfig`, `TrajectoryStore`, `ExperienceReplayBuffer`, `NeuralPatternStore`.
- `src/config.rs` — `SonaConfig::new(data_dir, dim)` (+ replay capacity, trajectory window).
- `src/trajectory.rs` — `TrajectoryStore` (sequences of state embeddings).
- `src/experience.rs` — `ExperienceReplayBuffer` (circular buffer of `(state, action, reward, next_state)`).
- `src/pattern.rs` — `NeuralPatternStore` (recognised patterns + confidence; searchable by category or similarity).

## Related

- `../../rvf-runtime`, `../../rvf-types`
- Sibling adapters under `../`
