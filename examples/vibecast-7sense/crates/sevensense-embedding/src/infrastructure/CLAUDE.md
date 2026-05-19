# sevensense-embedding/src/infrastructure

Infrastructure adapters for the embedding bounded context.

## Files
- `mod.rs` - Adapter wiring.
- `model_manager.rs` - Loads and caches Perch 2.0 ONNX models (downloads, version pinning).
- `onnx_inference.rs` - ONNX Runtime (`ort`) inference path with batching.

## Related
- Application services: `../application/services.rs`.
- Domain traits: `../domain/repository.rs`.
