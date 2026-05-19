# ruvector-router-core

High-performance vector database and neural routing inference engine. Provides vector storage/retrieval, HNSW indexing, multiple
quantization techniques (scalar/product/binary), SIMD distance kernels, and an AgenticDB-compatible API.

## Files

- `Cargo.toml` — `crate-type = ["lib", "staticlib"]` (also usable as a C-linkable archive). Depends on redb, memmap2, rayon,
  crossbeam, parking_lot, rkyv, bincode, serde, simsimd, ndarray, rand, uuid, chrono. Single criterion bench `vector_search`.
  `README.md` referenced.
- `src/lib.rs` — public API root.
- `benches/vector_search.rs` — HNSW + distance benchmark.

## Public API surface

Re-exported from `lib.rs`:
- `error::{Result, VectorDbError}`
- `types::{DistanceMetric, SearchQuery, SearchResult, VectorEntry}`
- `vector_db::VectorDB`

Internal modules also exposed: `distance`, `index`, `quantization`, `storage`.

## Related

- `../ruvector-core`, `../ruvector-index`, `../ruvector-quantization` — fuller vector-database stack.
- `../ruvector-diskann` — disk-resident ANN alternative.
- AgenticDB API consumers (router runtimes).
