# ruvector-hailo/src

Source root. Lots of conditional compilation based on `hailo` and `cpu-fallback` features.

## Always-available modules

- `lib.rs` — re-exports + `EmbeddingProvider` impl for `HailoEmbedder`.
- `device.rs` — `HailoDevice` (HailoRT handle wrapper).
- `error.rs` — `HailoError` (default-build calls return `Err(FeatureDisabled)`).
- `tokenizer.rs` — `WordPieceTokenizer`, `EncodedInput`, `SpecialIds`.
- `inference.rs` — `EmbeddingPipeline`, `mean_pool`, `l2_normalize`, `DEFAULT_MAX_SEQ`, `MINI_LM_DIM`.
- `hef_verify.rs` — HEF artifact sha256 pin verification.

## Feature-gated modules

- `hef_pipeline.rs` — real NPU pipeline (feature `hailo`).
- `hef_embedder.rs`, `hef_embedder_pool.rs` — NPU embedder + pool (features `hailo` + `cpu-fallback`).
- `cpu_embedder.rs`, `host_embeddings.rs` — candle/BERT-6 CPU fallback (feature `cpu-fallback`).
