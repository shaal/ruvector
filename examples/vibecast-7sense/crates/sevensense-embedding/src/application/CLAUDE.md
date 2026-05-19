# sevensense-embedding/src/application

Application layer for the embedding bounded context.

## Files
- `mod.rs` - Re-exports.
- `services.rs` - `EmbeddingService` orchestrating the ONNX model manager + normalization + quantization to convert audio segments into stored embeddings; supports batch processing.

## Related
- Domain types: `../domain/`.
- Adapters: `../infrastructure/`.
