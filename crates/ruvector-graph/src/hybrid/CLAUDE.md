# ruvector-graph/src/hybrid

Vector-Graph hybrid query system — combines vector similarity search with graph traversal for AI workloads (semantic search, RAG, GNN inference).

## Files

- `mod.rs` — Re-exports `SimilarityPredicate`, `VectorCypherExecutor`, `VectorCypherParser`, `GnnConfig`, `GraphEmbedding`, `GraphNeuralEngine`, `LinkPrediction`, `NodeClassification`, `Context`, `Evidence`, `RagConfig`, `RagEngine`, `ReasoningPath`, `ClusterResult`, `SemanticPath`, `SemanticSearch`, `SemanticSearchConfig`, `EmbeddingConfig`, `HybridIndex`, `VectorIndexType`.
- `vector_index.rs` — `HybridIndex`, `VectorIndexType`, `EmbeddingConfig` (HNSW + property-index marriage).
- `semantic_search.rs` — `SemanticSearch`, `SemanticPath`, `ClusterResult`, `SemanticSearchConfig`.
- `rag_integration.rs` — `RagEngine`, `RagConfig`, `Context`, `Evidence`, `ReasoningPath`.
- `graph_neural.rs` — `GraphNeuralEngine`, `GnnConfig`, `GraphEmbedding`, `LinkPrediction`, `NodeClassification`.
- `cypher_extensions.rs` — Cypher-language extensions for vector predicates (`SimilarityPredicate`, `VectorCypherParser`, `VectorCypherExecutor`).
