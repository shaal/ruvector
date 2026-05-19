# ruvbot / tests / integration / ruvector

Tests that exercise the WASM-based RuVector bindings consumed by the
learning subsystem (ADR-006).

## Files
- `wasm-bindings.test.ts` - Validates the WASM embedder and HNSW
  index integration used by `learning/embeddings` and `learning/search`.
