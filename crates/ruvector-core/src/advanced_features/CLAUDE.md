# ruvector-core/src/advanced_features

Production-leaning extensions to basic HNSW/flat search. Each module is wired up in the sibling `advanced_features.rs`.

## Files

- `compaction.rs` — segment compaction / vacuum.
- `conformal_prediction.rs` — calibrated confidence intervals on search results.
- `diskann.rs` — DiskANN-style graph index for billion-scale corpora.
- `filtered_search.rs` — metadata-filtered ANN search.
- `graph_rag.rs` — graph-based RAG retrieval.
- `hybrid_search.rs` — dense + sparse hybrid retrieval.
- `matryoshka.rs` — Matryoshka representation learning (truncatable embeddings).
- `mmr.rs` — Maximal Marginal Relevance re-ranking.
- `multi_vector.rs` — multi-vector (ColBERT-style) retrieval.
- `opq.rs` — Optimized Product Quantization.
- `product_quantization.rs` — PQ training + encoding.
- `sparse_vector.rs` — SPLADE-style sparse vector handling.
