# ruvector-gnn/src

Core source modules for the RuVector GNN crate. Combines tensor primitives, GNN layers, training/optimizer machinery, and
out-of-core (mmap) cold-tier support.

## Files

- `lib.rs` — crate root + docs; re-exports the major modules.
- `tensor.rs` — `ndarray`-backed tensor type and elementary ops.
- `layer.rs` — GNN layer implementations on HNSW topology.
- `graphmae.rs` — Graph-MAE style masked autoencoder pretraining.
- `training.rs` — `Optimizer` and `OptimizerType` (Adam with momentum + bias correction, etc.).
- `replay.rs` — `ReplayBuffer` with reservoir sampling for experience replay (continual learning).
- `ewc.rs` — `ElasticWeightConsolidation` to mitigate catastrophic forgetting.
- `scheduler.rs` — `LearningRateScheduler` / `SchedulerType` (warmup, plateau detection, etc.).
- `search.rs` — differentiable search over the GNN-augmented HNSW.
- `query.rs` — higher-level query primitives.
- `compress.rs` — model / graph compression utilities.
- `cold_tier.rs` — hyperbatch training pipeline for graphs exceeding RAM (gated by `cold-tier` feature).
- `mmap.rs`, `mmap_fixed.rs` — memory-mapped tensor backends (gated by `mmap` feature; non-WASM only).
- `error.rs` — crate-wide error enum.
