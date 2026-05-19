# ruvector-core/tests

Integration tests (`cargo test --test <name>`).

## Files

- `unit_tests.rs` — small isolated unit checks of internal helpers.
- `integration_tests.rs` — broad end-to-end VectorDB usage.
- `concurrent_tests.rs` — multi-threaded readers/writers.
- `stress_tests.rs` — high-load / long-running stress.
- `property_tests.rs` — proptest-based invariant checks.
- `hnsw_integration_test.rs` — HNSW insert/search correctness.
- `embeddings_test.rs` — `HashEmbedding` / `OnnxEmbedding` round-trips.
- `test_memory_pool.rs` — arena / pool behavior.
- `test_quantization.rs` — scalar/int4/PQ/binary quantization accuracy + speed.
- `test_simd_correctness.rs` — SIMD vs. scalar distance equivalence.
- `advanced_features_integration.rs` — DiskANN, filtered search, hybrid search, MMR, conformal prediction, etc.
