# cognitive_demo/src/components

The five components of the demo cognitive pipeline.

## Files

- `mod.rs` — re-exports each component.
- `sensor_adapter.rs` — `SensorAdapter`: ingests sensor input (Immutable region producer).
- `feature_extractor.rs` — `FeatureExtractor`: AppendOnly region writer for derived features.
- `reasoning_engine.rs` — `ReasoningEngine`: consumes features and emits proof-gated vector/graph mutations.
- `attestor.rs` — `Attestor`: produces attestations for downstream verification.
- `coordinator.rs` — `Coordinator`: drives `task_spawn`, `cap_grant`, `timer_wait` to orchestrate the others.
