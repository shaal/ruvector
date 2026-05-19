# sevensense-embedding/src

Source for the embedding bounded context.

## Files
- `lib.rs` - Crate root; documents DDD layout and re-exports `EmbeddingService`, `ModelManager`, `ModelConfig`.
- `normalization.rs` - Vector normalization helpers (L2, mean/variance).
- `quantization.rs` - Quantization helpers (int8 / fp16) used when storing or transporting embeddings.

## Subdirectories
- `domain/` - `Embedding`, `EmbeddingModel`, repository traits.
- `application/` - Embedding/batch services.
- `infrastructure/` - ONNX Runtime integration, model manager.

## Related
- Parent: `../CLAUDE.md`.
