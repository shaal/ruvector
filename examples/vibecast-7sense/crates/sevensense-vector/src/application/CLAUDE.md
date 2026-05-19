# sevensense-vector/src/application

Application layer for the vector bounded context.

## Files
- `mod.rs` - Re-exports.
- `services.rs` - `VectorSpaceService` orchestrating HNSW indexing, similarity search, filtering, batch ops, and persistence.

## Related
- Domain types: `../domain/`.
- Adapters: `../infrastructure/`.
