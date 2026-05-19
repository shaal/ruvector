# ruvector-gnn

Graph Neural Network layer for Ruvector on HNSW topology. Provides tensor operations, GNN layers, online/batch training,
compression, differentiable search, and (per Issue #17) continual-learning anti-forgetting machinery (Adam + replay buffer + EWC +
LR scheduler).

## Files

- `Cargo.toml` — depends on `ruvector-core`, `ndarray 0.17`, rand/rand_distr, rayon, dashmap, parking_lot, optional `memmap2`
  and `napi`. Linux gets `libc`. Dev: criterion, proptest, tempfile.
- `README.md` — public documentation.
- `src/` — library code (see `src/CLAUDE.md`).
- `tests/` — integration tests (`loss_demo.rs`, `loss_verification.rs`).
- `examples/` — runnable examples (currently empty placeholder dir).

## Features

- `default = ["simd", "mmap"]`.
- `simd` — enable SIMD distance / matmul paths.
- `wasm` — WASM-compatible build (drops mmap/page_size).
- `napi` — N-API bindings (`napi`, `napi-derive`).
- `mmap` — memory-mapped tensors (`memmap2` + `page_size`).
- `cold-tier` — implies `mmap`, enables hyperbatch training for graphs exceeding RAM.

## Public API surface

Documented modules include `training::{Optimizer, OptimizerType}`, `replay::ReplayBuffer`, `ewc::ElasticWeightConsolidation`,
`scheduler::{LearningRateScheduler, SchedulerType}`, plus `layer`, `graphmae`, `search`, `query`, `compress`, `cold_tier`,
`mmap`, `tensor`, `error`.

## Related

- `../ruvector-core` — base HNSW / index types.
- `../ruvector-attention`, `../ruvector-quantization` — complementary model components.
- `../../npm/packages/ruvector-gnn` (if present) — corresponding npm wrapper.
