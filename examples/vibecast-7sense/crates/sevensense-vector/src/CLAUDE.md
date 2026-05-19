# sevensense-vector/src

Source for the vector bounded context.

## Files
- `lib.rs` - Crate root; documents DDD layout and re-exports public types.
- `distance.rs` - Distance metric implementations (Euclidean, cosine, inner product, etc.).
- `hyperbolic.rs` - Poincare ball model helpers for hierarchical embeddings.

## Subdirectories
- `domain/` - `EmbeddingId`, `HnswConfig`, `SimilarityEdge`, repository traits, error type.
- `application/` - `VectorSpaceService` use cases.
- `infrastructure/` - Local HNSW index, graph edge store.

## Related
- Parent: `../CLAUDE.md`.
