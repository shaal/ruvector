# OSpipe / src / learning

Learning loop module: closes feedback from search / user-interaction back into stored embeddings, rerankers, or graph weights.

## Important files
- `mod.rs` - module root. Currently the sole entry; defines the learning loop API used by the server and pipeline.

## Related
- Reranker: `../search/reranker.rs`. Vector store: `../storage/vector_store.rs`.
- Higher-level learning patterns live in `../../../ruvLLM/src/sona/` (SONA / ReasoningBank).
