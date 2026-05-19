# sevensense-vector

Vector-database operations and HNSW indexing for 7sense: local HNSW index (claimed 150x speedup over brute force), optional Qdrant client wrapper, hyperbolic embeddings for hierarchical relationships, persistence, batch operations.

## Files
- `Cargo.toml` - Depends on `sevensense-core`, `tokio`, `async-trait`, vector / HNSW deps.
- `src/lib.rs` - Crate root and architecture overview.
- `src/distance.rs` - Distance metrics (Euclidean / cosine / inner product / hyperbolic).
- `src/hyperbolic.rs` - Poincare ball hyperbolic embedding helpers.
- `src/domain/` - Entities, repository traits, error types.
- `src/application/` - `VectorSpaceService`.
- `src/infrastructure/` - Local HNSW index, graph edge store.
- `benches/hnsw_benchmark.rs` - Criterion HNSW benchmarks.

## Build / bench
```
cargo build -p sevensense-vector
cargo bench -p sevensense-vector
```

## Related
- Consumed by `sevensense-learning`, `sevensense-analysis`, `sevensense-api`.
