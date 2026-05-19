# ruvector-delta-index

Delta-aware HNSW index with incremental updates and repair strategies. Optimised for workloads with frequent small changes to vector embeddings: apply a `VectorDelta`, monitor recall, and run targeted graph-repair passes instead of rebuilding the whole index.

## Features

- `default = ["parallel"]`
- `parallel = ["rayon"]`
- `simd = ["simsimd"]`
- `persistence = ["bincode"]`

## Layout

- `Cargo.toml` — path dep on `ruvector-delta-core` (provides `Delta`, `DeltaStream`, `VectorDelta`); other deps: `priority-queue`, `dashmap`, `parking_lot`, `smallvec`, `thiserror`, `rand`, `rand_xorshift`.
- `src/lib.rs` — crate root, defines `DeltaHnswConfig`, the `DeltaHnsw` index, re-exports submodules.
- `src/incremental.rs` — `IncrementalUpdater` (apply deltas).
- `src/repair.rs` — `GraphRepairer`, `RepairConfig`, `RepairStrategy`.
- `src/quality.rs` — `QualityMonitor`, `QualityMetrics`, `RecallEstimate`.
- `src/error.rs` — `IndexError`, `Result`.

## Public API

`DeltaHnsw::new(dim, config)`, `insert(id, vec)`, `apply_delta(id, &VectorDelta)`, `search(query, k)`; configuration via `DeltaHnswConfig`; repair via `GraphRepairer` + `RepairStrategy`.

## Related

- `crates/ruvector-delta-core` — `Delta`, `DeltaStream`, `VectorDelta`.
- `crates/ruvector-hnsw` (or sibling HNSW crates) — full-rebuild HNSW baselines.
- `crates/micro-hnsw-wasm` — sibling tiny HNSW for WASM.
