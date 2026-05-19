# ruvector-tiny-dancer-core

Tiny Dancer: production-grade AI agent routing system. Routes LLM requests to
candidate models to minimize cost using a FastGRNN neural router (sub-millisecond
latency), feature engineering, model optimization (quantization, pruning),
conformal-prediction uncertainty quantification, circuit-breaker degradation, and
SQLite/AgentDB persistence. Includes training infrastructure with knowledge
distillation.

## Layout

- `Cargo.toml` — `crate-type = ["lib", "staticlib"]`. Deps: redb, memmap2, rayon,
  crossbeam, parking_lot, simsimd, ndarray, rusqlite (bundled, modern_sqlite),
  bytemuck, dashmap, chrono, uuid. Dev: criterion, proptest, tempfile.
- `src/lib.rs` — module roots; re-exports `FastGRNN`/`FastGRNNConfig`, `Router`,
  trainer types, routing DTOs.
- `src/model.rs` — `FastGRNN` neural router.
- `src/router.rs` — `Router` integrating model + features + circuit breaker.
- `src/feature_engineering.rs` — per-candidate feature vectors.
- `src/optimization.rs` — quantization / pruning.
- `src/uncertainty.rs` — conformal prediction.
- `src/circuit_breaker.rs` — degradation patterns.
- `src/storage.rs` — SQLite/redb persistence.
- `src/training.rs` — `Trainer`, `TrainingConfig`, `TrainingDataset`,
  `TrainingMetrics`, teacher prediction generator (knowledge distillation).
- `src/api.rs` — HTTP admin/training API surface.
- `src/tracing.rs`, `src/metrics.rs` — observability.
- `src/types.rs` — `Candidate`, `RouterConfig`, `RoutingRequest`/`Response`,
  `RoutingDecision`, `RoutingMetrics`.
- `src/error.rs` — `TinyDancerError`, `Result`.
- `benches/feature_engineering.rs`, `benches/routing_inference.rs` — Criterion.
- `examples/admin-server.rs`, `examples/train-model.rs`,
  `examples/metrics_example.rs`, `examples/tracing_example.rs`,
  `examples/full_observability.rs` (+ `OBSERVABILITY_EXAMPLES.md`).
- `docs/` — API reference, observability, training guides.

## Public API

`FastGRNN`, `FastGRNNConfig`, `Router`, `Trainer`, `TrainingConfig`,
`TrainingDataset`, `TrainingMetrics`, `generate_teacher_predictions`,
`Candidate`, `RouterConfig`, `RoutingDecision`, `RoutingMetrics`,
`RoutingRequest`, `RoutingResponse`, `TinyDancerError`, `Result`, `VERSION`.

## Related

Used by higher-level neural-routing services in the workspace.
