# rvf-adapter-agentdb/src

Source.

## Files

- `lib.rs` — re-exports the three sub-modules; docs the VEC_SEG/INDEX_SEG/META_SEG mapping.
- `vector_store.rs` — `RvfVectorStore` (embedding rows ↔ VEC_SEG).
- `index_adapter.rs` — `RvfIndexAdapter` (HNSW A/B/C layers ↔ INDEX_SEG).
- `pattern_store.rs` — `RvfPatternStore` + `MemoryPattern` (reward/critique/tag metadata ↔ META_SEG).
