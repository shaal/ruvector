# ruvbot / src / learning

Learning bounded context (ADR-007). Holds the self-learning pipeline:
embeddings, memory management, pattern detection, hybrid search, and
training. Reachable via the package subpath export `ruvbot/learning`.

## Files
- `index.ts` - Barrel re-exporting the submodules below.

## Subdirectories
- `embeddings/` - WASM-backed text embedder.
- `memory/` - `MemoryManager` for short/long-term recall.
- `patterns/` - Pattern extraction and clustering.
- `search/` - Hybrid BM25 + HNSW retrieval.
- `training/` - Online training loop for personalization.
