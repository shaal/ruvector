# ruvector-delta-index/src

- `lib.rs` — crate root, declares `DeltaHnswConfig`, the `DeltaHnsw` index struct, re-exports `IncrementalUpdater`, `QualityMetrics`, `QualityMonitor`, `RecallEstimate`, `GraphRepairer`, `RepairConfig`, `RepairStrategy`, `IndexError`, `Result`.
- `incremental.rs` — `IncrementalUpdater` that consumes `Delta` / `DeltaStream` from `ruvector-delta-core` and patches the graph.
- `repair.rs` — graph-quality repair pipeline (`GraphRepairer`, `RepairStrategy` enum, `RepairConfig`).
- `quality.rs` — `QualityMonitor`, `QualityMetrics`, `RecallEstimate` for online recall tracking.
- `error.rs` — `thiserror`-based `IndexError`.

See `../CLAUDE.md`.
