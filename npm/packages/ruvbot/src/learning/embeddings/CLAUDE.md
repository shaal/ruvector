# ruvbot / src / learning / embeddings

Embedding generation. Wraps the WASM model from RuVector to produce
vectors used by the hybrid search and memory subsystems (ADR-006).

## Files
- `WasmEmbedder.ts` - Loads the WASM embedding module, exposes
  `embed(text)` and batch helpers; handles initialization caching.
- `index.ts` - Barrel re-exporting `WasmEmbedder`.
