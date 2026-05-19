# ruvector-tiny-dancer-core/src

Source for the Tiny Dancer routing system.

- `lib.rs` — module roots, public re-exports, `VERSION`.
- `model.rs` — `FastGRNN` neural router + `FastGRNNConfig`.
- `router.rs` — `Router` ties together model, feature engineering, circuit
  breaker, and storage.
- `feature_engineering.rs` — per-candidate feature vectors (simsimd-aware).
- `optimization.rs` — model quantization / pruning helpers.
- `uncertainty.rs` — conformal prediction for confidence intervals.
- `circuit_breaker.rs` — graceful-degradation circuit breaker.
- `storage.rs` — SQLite (`rusqlite`) + redb persistence.
- `training.rs` — `Trainer`, `TrainingConfig`, `TrainingDataset`,
  `TrainingMetrics`, `generate_teacher_predictions` (knowledge distillation).
- `api.rs` — HTTP admin/training API surface.
- `tracing.rs`, `metrics.rs` — observability.
- `types.rs` — `Candidate`, `RouterConfig`, `RoutingDecision`, `RoutingMetrics`,
  `RoutingRequest`, `RoutingResponse`.
- `error.rs` — `TinyDancerError` + `Result`.
