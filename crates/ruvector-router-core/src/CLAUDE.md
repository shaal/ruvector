# ruvector-router-core/src

Vector DB + neural routing inference engine source.

## Files

- `lib.rs` — module declarations and re-exports.
- `types.rs` — `DistanceMetric`, `SearchQuery`, `SearchResult`, `VectorEntry` — canonical public types.
- `error.rs` — `Result` alias and `VectorDbError`.
- `vector_db.rs` — `VectorDB` top-level facade orchestrating storage + index + quantization.
- `index.rs` — HNSW (Hierarchical Navigable Small World) index implementation.
- `distance.rs` — SIMD-optimized distance kernels (via `simsimd`).
- `quantization.rs` — scalar / product / binary quantization codecs.
- `storage.rs` — persistent storage layer (redb / mmap-backed).
