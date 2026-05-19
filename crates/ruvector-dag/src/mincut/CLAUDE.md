# ruvector-dag/src/mincut

Sub-polynomial bottleneck detection and optimisation over query DAGs (O(n^0.12) target per crate-level docs).

- `mod.rs` — re-exports.
- `engine.rs` — `DagMinCutEngine`, `MinCutConfig`, `MinCutResult`, `FlowEdge`.
- `bottleneck.rs` — `Bottleneck`, `BottleneckAnalysis` summary types.
- `local_kcut.rs` — `LocalKCut` (local k-cut variant for incremental analysis).
- `dynamic_updates.rs` — incremental edge-add/remove handling.
- `redundancy.rs` — `RedundancyStrategy`, `RedundancySuggestion` — propose redundancy to relieve bottlenecks.

Public surface re-exported from `../lib.rs`. See `../CLAUDE.md`.
