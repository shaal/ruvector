# ruvector-dag/src/healing

Self-healing subsystem for neural DAG learning. Requires `feature = "full"`.

- `mod.rs` — module wiring + re-exports.
- `anomaly.rs` — `Anomaly`, `AnomalyConfig`, `AnomalyDetector`, `AnomalyType`.
- `drift_detector.rs` — `LearningDriftDetector`, `DriftMetric`, `DriftTrend`.
- `index_health.rs` — `IndexHealth`, `IndexHealthChecker`, `IndexThresholds`, `IndexType`, `IndexCheckResult`, `HealthStatus`.
- `orchestrator.rs` — `HealingOrchestrator`, `HealingCycleResult` — runs the detect → diagnose → repair loop.
- `strategies.rs` — `RepairStrategy` trait + implementations (`CacheFlushStrategy`, `IndexRebalanceStrategy`, `PatternResetStrategy`), `RepairResult`.

See `../CLAUDE.md` and `examples/self_healing.rs`.
