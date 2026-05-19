# ruvector-core

High-performance Rust-native vector database core. The foundational crate of the RuVector workspace: HNSW indexing, SIMD distance (SimSIMD), quantization (scalar / int4 / PQ / binary), REDB persistence, memory-mapped storage, and assorted "advanced" search features.

## Important files

- `Cargo.toml` — many optional deps (`redb`, `memmap2`, `hnsw_rs`, `simsimd`, `rayon`, `crossbeam`, `rkyv`, `bincode`). Numerous feature flags gate optional behavior.
- `src/lib.rs` — public re-exports; documents working vs. experimental features and the AgenticDB placeholder-embedding warning.
- `README.md` — user-facing intro / quickstart.
- `docs/EMBEDDINGS.md` — guidance on hash vs. ONNX embeddings.

## Module map (src/)

- `vector_db.rs` — top-level `VectorDB` façade.
- `agenticdb.rs` — agent-oriented DB layer (uses placeholder embeddings by default; see lib.rs warning).
- `types.rs`, `error.rs` — shared types and error enum.
- `index.rs`, `index/` — index trait + implementations (`flat.rs`, `hnsw.rs`).
- `storage.rs`, `storage_compat.rs`, `storage_memory.rs` — REDB / mmap / in-memory backends.
- `distance.rs`, `simd_intrinsics.rs` — distance metrics (cosine, L2, IP) with SimSIMD + portable fallbacks.
- `quantization.rs` — scalar (4x), int4 (8x), PQ (8-16x), binary (32x).
- `memory.rs`, `arena.rs`, `cache_optimized.rs`, `lockfree.rs` — perf primitives.
- `embeddings.rs` — hash / ONNX embedders (see `examples/embeddings_example.rs`).
- `advanced/` — hypergraph, learned index, neural hash, TDA (research features).
- `advanced_features/` — production-leaning extras (DiskANN, compaction, conformal prediction, filtered search, hybrid search, Matryoshka, MMR, multi-vector, OPQ, PQ, sparse vector, graph RAG).

## Tests & benches

- `tests/` — unit, integration, concurrent, stress, property, HNSW integration, memory pool, quantization, SIMD correctness, embeddings, advanced features.
- `benches/` — batch operations, memory, SIMD, comprehensive, distance metrics, HNSW search, quantization, real-world bench.
- `fuzz/` — cargo-fuzz harness; `fuzz_targets/fuzz_distance.rs`.
- `examples/` — `embeddings_example.rs`, `neon_benchmark.rs`.

## Public API surface

`VectorDB`, `AgenticDB`, `Index`/`FlatIndex`/`HnswIndex`, `Storage` traits, `Distance`/`DistanceMetric`, `Quantization` types, `Embedding`/`OnnxEmbedding`, advanced types under `advanced_features::`.

## Related

- Used by virtually every other ruvector crate (snapshot, raft, mincut, etc.).
- Optional ONNX feature `onnx-embeddings` for real semantic embeddings.
