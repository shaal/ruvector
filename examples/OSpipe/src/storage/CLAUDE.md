# OSpipe / src / storage

Storage layer for OSpipe. Defines the vector-store trait, an embedding interface, and shared persistence traits used by the pipeline.

## Important files
- `mod.rs` - module root and public storage API.
- `traits.rs` - core traits (`VectorStore`, embedding provider, persistence) abstracted over backends.
- `vector_store.rs` - concrete vector-store implementation backed by `ruvector-core` / `ruvector-filter` / `ruvector-cluster`.
- `embedding.rs` - embedding provider abstraction and helpers.

## Related
- Upstream: `../pipeline/`. Downstream: `../search/`.
- Persisted via `../persistence.rs` (parent module).
