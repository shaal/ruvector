# sevensense-vector/src/infrastructure

Infrastructure adapters for the vector bounded context.

## Files
- `mod.rs` - Adapter wiring.
- `hnsw_index.rs` - Local HNSW index implementation (150x faster than brute force per crate docs).
- `graph_store.rs` - Persisted similarity-graph edges (backing `GraphEdgeRepository`).

## Related
- Application services: `../application/services.rs`.
- Bench: `../../benches/hnsw_benchmark.rs`.
