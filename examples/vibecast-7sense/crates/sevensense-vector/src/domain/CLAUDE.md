# sevensense-vector/src/domain

Domain layer of the vector bounded context.

## Files
- `mod.rs` - Aggregates and re-exports.
- `entities.rs` - `EmbeddingId`, `HnswConfig`, `SimilarityEdge`.
- `repository.rs` - `VectorIndexRepository`, `GraphEdgeRepository` traits.
- `error.rs` - `VectorError` type.

## Related
- Application layer: `../application/`.
- Infrastructure: `../infrastructure/`.
