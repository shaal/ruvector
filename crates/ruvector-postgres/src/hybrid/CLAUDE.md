# ruvector-postgres/src/hybrid

Hybrid Search (BM25 + Vector) — combined keyword + semantic vector search with multiple fusion strategies (RRF default, linear blend, learned/adaptive). Vector and keyword branches run concurrently.

## Files

- `mod.rs` — Module entry with SQL interface documentation.
- `bm25.rs` — BM25 scoring with document-length normalization.
- `fusion.rs` — Fusion algorithms (RRF, linear, learned).
- `executor.rs` — Parallel execution of vector + BM25 branches.
- `registry.rs` — Tracks hybrid-enabled collections with per-collection settings.
- `tests.rs` — Inline tests.
