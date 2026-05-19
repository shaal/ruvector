# ruvector-hailo

Ruvector embedding backend for the Hailo-8 NPU (ADR-167). Implements `ruvector_core::embeddings::EmbeddingProvider` (iter 218 closed ADR-178 Gap B by landing the path dep + impl block).

## Build modes

- Default build (no `hailo` feature) — every API call returns `Err(HailoError::FeatureDisabled)`. Lets non-Pi machines `cargo check -p ruvector-hailo` without HailoRT installed.
- `hailo` feature — activates the real Hailo NPU runtime via `hailort-sys`.
- `cpu-fallback` feature — adds candle / tokenizers so `HailoEmbedder::open` falls back to `CpuEmbedder` when no `model.hef` is present (real semantic vectors on Cortex-A76 NEON / x86 AVX2; NPU stays idle until HEF is supplied).

`publish = false` (internal crate; path dep on the in-tree `hailort-sys` FFI binding makes it non-publishable anyway).

## Layout

- `Cargo.toml` — features + dep set. Optional sha256 pinning via `RUVECTOR_HEF_SHA256` env var (used only under `hailo` feature in `hef_pipeline.rs`).
- `deny.toml` — cargo-deny config (allow-wildcard-paths exception for in-workspace deps).
- `src/lib.rs` — public modules + re-exports of `HailoDevice`, `HailoError`, `EmbeddingPipeline`, `l2_normalize`, `mean_pool`, `DEFAULT_MAX_SEQ`, `MINI_LM_DIM`, `WordPieceTokenizer`, `EncodedInput`, `SpecialIds`.
- `src/device.rs` — `HailoDevice` (HailoRT handle wrapper).
- `src/error.rs` — `HailoError`.
- `src/tokenizer.rs` — `WordPieceTokenizer`, `EncodedInput`, `SpecialIds`.
- `src/inference.rs` — `EmbeddingPipeline`, `mean_pool`, `l2_normalize`, constants.
- `src/hef_verify.rs` — HEF artifact verification (sha256-pin check).
- `src/hef_pipeline.rs` (feature `hailo`) — real NPU pipeline that loads HEFs.
- `src/hef_embedder.rs` + `src/hef_embedder_pool.rs` (features `hailo` + `cpu-fallback`) — NPU-backed embedder with CPU fallback + pool.
- `src/cpu_embedder.rs`, `src/host_embeddings.rs` (feature `cpu-fallback`) — candle/sentence-transformers BERT-6 path.

## Tests / benches / models

- `benches/wordpiece_throughput.rs` — tokenizer throughput.
- `tests/cpu_fallback_integration.rs` — end-to-end CPU fallback path.
- `tests/tokenizer_proptest.rs` — proptest invariants on WordPiece.
- `models/` — directory for `model.hef` / `model.safetensors` / `tokenizer.json` artifacts (gitignored at runtime).

## Related crates

- `crates/hailort-sys` — in-tree FFI binding to Hailo's C runtime.
- `crates/ruvector-core` — provides the `EmbeddingProvider` trait being implemented.
