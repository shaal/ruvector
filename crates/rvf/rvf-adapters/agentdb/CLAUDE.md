# rvf-adapter-agentdb

RVF adapter for AgentDB. Maps agentdb's vector storage, HNSW index, and memory-pattern APIs onto the RVF segment model:

- **VEC_SEG**: raw vector data (episodes, state embeddings)
- **INDEX_SEG**: HNSW index layers (A/B/C progressive)
- **META_SEG**: memory-pattern metadata (rewards, critiques, tags)

Uses the RVText domain profile (text/embedding workloads).

## Layout

- `Cargo.toml` — name `rvf-adapter-agentdb`. Deps: `rvf-runtime`, `rvf-types`, `rvf-index` (all with `std` feature). Dev: `tempfile`.
- `src/lib.rs` — module decls + re-exports of `RvfIndexAdapter`, `MemoryPattern`, `RvfPatternStore`, `RvfVectorStore`.
- `src/vector_store.rs` — `RvfVectorStore` (VEC_SEG-backed embedding store).
- `src/index_adapter.rs` — `RvfIndexAdapter` (HNSW layers over INDEX_SEG).
- `src/pattern_store.rs` — `MemoryPattern` + `RvfPatternStore` (META_SEG records).

## Related

- `../../rvf-runtime`, `../../rvf-types`, `../../rvf-index`
- Sibling adapters: `../claude-flow`, `../agentic-flow`, `../ospipe`, `../rvlite`, `../sona`
