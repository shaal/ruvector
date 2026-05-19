# ruvllm_sparse_attention

Subquadratic O(N log N) sparse-attention kernel for Rust LLM inference on edge devices, with optional FastGRNN salience gating for near-linear O(N) scaling. Designed to run on ESP32-S3 / RISC-V MCUs with `no_std + alloc`, and with full parallel `rayon` performance on multicore servers when `std` is enabled.

## Features

- `default = ["std"]`
- `std` — standard library (disable for no_std + alloc).
- `parallel = ["std", "dep:rayon"]` — parallel head loops (~4x prefill speedup).
- `fp16 = ["dep:half"]` — FP16 KV cache (halves KV memory; works in no_std).

## Layout

- `Cargo.toml` — minimal runtime deps. `libm` (always) for f32 transcendentals on no_std; optional `rayon`, `half`. `rand` only used in tests/benches (zero runtime footprint, ADR-183).
- `src/` — five files; see `src/CLAUDE.md`.
- `benches/` — `attention_bench.rs`, `sparse_mario_bench.rs`.
- `examples/` — `esp32s3_smoke.rs`, `fastgrnn_gated_scaling.rs`, `run_sparse_attention.rs`, `sparse_mario.rs`.
- `docs/` — `TUTORIAL.md`, `sparse_mario_baselines.md`, `sparse_mario_metrics.md`, `benchmark_edge_estimates.csv`, `adr/ADR_0001_subquadratic_sparse_attention.md`.

## Public API

`dense_attention`, `AttentionBackend`, `AttentionError`, `IncrementalLandmarks`, `KvCache`, `SparseAttentionConfig`, `SubquadraticSparseAttention`; `FastGrnnGate` + `FASTGRNN_DEFAULT_HIDDEN_DIM`; `RuvLlmSparseBlock`, `RuvLlmSparseBlockConfig`; `Tensor3`. With `fp16`: `KvCacheF16`. Under no_std, `no_std_math::F32Ext` provides `.exp() / .sqrt() / .tanh() / .powi()` on `f32` via `libm`.

## Related

- `crates/ruvllm` — broader ruvLLM engine (LLM runtime that hosts this attention kernel).
- `crates/ruvector-hailo-cluster` — `ruvllm-engine` feature uses ruvLLM (and thus this kernel) on the Pi worker.
- `examples/esp32s3_smoke.rs` — embedded smoke test for the no_std build.
