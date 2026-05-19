# ruvector-graph/src/optimization

Performance optimization modules targeting 100x speedup over Neo4j.

## Files

- `mod.rs` — Module declarations.
- `simd_traversal.rs` — SIMD-vectorized graph traversal.
- `bloom_filter.rs` — Bloom filters for negative lookups.
- `cache_hierarchy.rs` — Cache-optimized data layouts.
- `memory_pool.rs` — Custom memory allocator/pool.
- `index_compression.rs` — Compressed index encodings.
- `adaptive_radix.rs` — Adaptive Radix Tree for property indexes.
- `query_jit.rs` — JIT-compiled query operators.

## Pointers

- Consumed by `../executor/operators.rs` and `../executor/pipeline.rs`.
