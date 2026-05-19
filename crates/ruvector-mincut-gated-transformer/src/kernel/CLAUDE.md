# ruvector-mincut-gated-transformer/src/kernel

Low-level numeric kernels.

## Files

- `mod.rs` — re-exports.
- `qgemm.rs` — quantised GEMM (INT4 / Q15 paths).
- `quant4.rs` — INT4 pack / unpack (`int4` feature).
- `norm.rs` — RMSNorm (`rmsnorm` feature).
- `bench_utils.rs` — internal helpers shared by `benches/kernel.rs`.
