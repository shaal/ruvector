# ruvector-core/src

Implementation of the vector DB core.

## Top-level façades

- `lib.rs` — module declarations, re-exports, feature-warning docs.
- `vector_db.rs` — `VectorDB` (canonical user-facing handle).
- `agenticdb.rs` — `AgenticDB` agent-oriented layer (uses placeholder embeddings unless configured otherwise).

## Indexes

- `index.rs` — index trait dispatch.
- `index/flat.rs` — exact-search flat index.
- `index/hnsw.rs` — HNSW approximate index (via `hnsw_rs`).

## Storage

- `storage.rs` — primary `Storage` trait + REDB-backed impl.
- `storage_compat.rs` — backward-compat wrappers.
- `storage_memory.rs` — pure in-memory backend.

## Distance / quantization

- `distance.rs` — `DistanceMetric` enum + portable kernels.
- `simd_intrinsics.rs` — SimSIMD / AVX2 / NEON acceleration.
- `quantization.rs` — scalar (4x), int4 (8x), PQ (8-16x), binary (32x) plus quantized distances.

## Perf primitives

- `arena.rs`, `memory.rs`, `cache_optimized.rs`, `lockfree.rs` — bump arena, custom allocators, cache-friendly layouts, lock-free structures.

## Embeddings

- `embeddings.rs` — `HashEmbedding` (placeholder) and `OnnxEmbedding` (real semantic embeddings via `onnx-embeddings` feature).

## Shared

- `types.rs` — `Vector`, `Id`, config types.
- `error.rs` — crate error enum.

## Research / extended

- `advanced/` — hypergraph, learned index, neural hash, TDA.
- `advanced_features.rs` + `advanced_features/` — DiskANN, conformal prediction, hybrid/filtered search, Matryoshka, MMR, multi-vector, OPQ, PQ, sparse vector, GraphRAG, compaction.
