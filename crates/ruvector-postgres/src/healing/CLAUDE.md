# ruvector-postgres/src/healing

Self-Healing Engine — automated problem detection and remediation: integrity monitoring -> detector -> strategy execution (with rollback) -> learning loop.

## Files

- `mod.rs` — Module entry + architecture doc.
- `detector.rs` — `ProblemDetector`: monitors health, detects state transitions (normal/stress/critical).
- `engine.rs` — `RemediationEngine`: orchestrates strategy execution with rollback.
- `strategies.rs` — Library of remediation strategies.
- `learning.rs` — Tracks outcomes to improve strategy selection over time.
- `worker.rs` — Background worker for continuous monitoring.
- `functions.rs` — pgrx SQL functions to query/trigger the healing engine.
