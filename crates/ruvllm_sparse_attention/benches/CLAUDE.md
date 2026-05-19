# ruvllm_sparse_attention/benches

Criterion benchmarks.

- `attention_bench.rs` — head-to-head dense vs `SubquadraticSparseAttention` (and optional FastGRNN-gated) on synthetic prefill / decode workloads.
- `sparse_mario_bench.rs` — benchmark against the "sparse Mario" reference workload (see `../docs/sparse_mario_baselines.md`).

See `../CLAUDE.md`.
