# OSpipe / src / search

Search stage. Combines vector similarity, graph context, reranking, MMR diversification, and routing into the unified query surface used by the HTTP server.

## Important files
- `mod.rs` - module root and public search API.
- `hybrid.rs` - hybrid vector + lexical/graph retrieval.
- `enhanced.rs` - higher-level enhanced search that layers extra signals on top of hybrid.
- `reranker.rs` - second-stage reranker over candidate hits.
- `mmr.rs` - Maximal Marginal Relevance diversification.
- `router.rs` - query router (built on `ruvector-router-core`) that picks among search strategies.

## Related
- Storage backing the index: `../storage/`.
- Graph features: `../graph/`. Server endpoints: `../server/`.
