# ruvector-mincut-node/src

Single-file NAPI binding layer.

- `lib.rs` — `#[napi]` exports wrapping `ruvector_mincut` core types:
  - `JsEdge { id, source, target, weight }`
  - `JsStats { insertions, deletions, queries, avg_update_time_us }`
  - `MinCut` (over `DynamicMinCut` + `DynamicGraph`, built with `MinCutBuilder`).
  - `ThreeLevelHierarchy` (over `cluster::hierarchy::ThreeLevelHierarchy` with
    `HierarchyConfig`).
  - `LocalKCut` (over `localkcut::deterministic::DeterministicLocalKCut`).
  - `MinCutWrapper` (over `RustMinCutWrapper`).

Thread-safe wrappers use `Arc<Mutex<...>>` to satisfy NAPI's `Send + Sync`.
