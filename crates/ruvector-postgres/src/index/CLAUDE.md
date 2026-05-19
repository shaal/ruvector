# ruvector-postgres/src/index

Vector similarity-search index implementations for PostgreSQL.

- **HNSW**: Hierarchical Navigable Small World graphs for fast ANN search.
- **IVFFlat**: Inverted File with Flat quantization for scalable search (v2 supports quantization).

Access methods registered: `ruhnsw`, `ruivfflat`.

## Files

- `mod.rs` — Module entry; documents access-method registration + SQL usage.
- `hnsw.rs` — Core HNSW index data structures.
- `hnsw_am.rs` — pgrx access-method bindings for `ruhnsw`.
- `ivfflat.rs` — Core IVFFlat algorithm.
- `ivfflat_am.rs` — pgrx access-method bindings for `ruivfflat`.
- `ivfflat_storage.rs` — On-disk storage for IVFFlat.
- `scan.rs` — Index scan execution.
- `bgworker.rs` — Background worker support for index maintenance.
- `parallel.rs` / `parallel_ops.rs` — Parallel build/scan helpers.

## Pointers

- See `../../docs/QUICK_REFERENCE_IVFFLAT.md`, `../../docs/ivfflat_access_method.md`, `../../docs/SIMD_OPTIMIZATION.md`.
