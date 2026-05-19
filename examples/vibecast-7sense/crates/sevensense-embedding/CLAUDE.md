# sevensense-embedding

Embedding bounded context for 7sense: Perch 2.0 ONNX integration producing 1536-dimensional embeddings from audio segments. Handles model loading, inference, normalization, and quantization.

## Files
- `Cargo.toml` - Depends on `sevensense-core`, `sevensense-audio`, `ort` (ONNX Runtime), numerical libs.
- `src/lib.rs` - Crate root and architecture overview.
- `src/normalization.rs` - L2 / mean-variance normalization.
- `src/quantization.rs` - int8/fp16 quantization for storage efficiency.
- `src/domain/` - `Embedding`, `EmbeddingModel` entities, repository traits.
- `src/application/` - `EmbeddingService`, batch processing.
- `src/infrastructure/` - ONNX Runtime inference, model manager.

## Build
```
cargo build -p sevensense-embedding
```

## Related
- Consumes: `sevensense-audio`. Consumed by: `sevensense-vector`, `sevensense-learning`, `sevensense-api`.
- Benchmark: `../../benches/embedding_benchmark.rs`.
