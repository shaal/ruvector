# ruvbot / src / learning / search

Hybrid retrieval (ADR-009). Combines lexical BM25 scoring with HNSW
vector similarity to retrieve relevant memories and skills.

## Files
- `BM25Index.ts` - In-memory BM25 lexical index with configurable
  k1/b parameters.
- `HybridSearch.ts` - Fuses BM25 and vector results with a tunable
  weight; returns ranked candidates.
- `index.ts` - Barrel re-exporting both.
