# ruvbot / tests / integration / core

Cross-cutting integration tests for core capabilities.

## Files
- `bm25-index.test.ts` - Lexical index correctness.
- `byzantine-consensus.test.ts` - PBFT correctness under failures.
- `hybrid-search.test.ts` - Fusion of BM25 + HNSW results.
- `providers.test.ts` - LLM provider adapters end-to-end (mocked).
- `swarm-coordinator.test.ts` - Multi-agent orchestration.
