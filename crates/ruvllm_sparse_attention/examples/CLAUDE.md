# ruvllm_sparse_attention/examples

Runnable examples (`cargo run --example <name> -p ruvllm_sparse_attention`).

- `run_sparse_attention.rs` — minimal usage of `SubquadraticSparseAttention`.
- `fastgrnn_gated_scaling.rs` — turns on `FastGrnnGate` and shows the O(N) scaling regime.
- `sparse_mario.rs` — exercises the "sparse Mario" reference workload referenced in `../docs/`.
- `esp32s3_smoke.rs` — smoke test for the no_std + alloc path; build for an ESP32-S3 target.

See `../CLAUDE.md`.
