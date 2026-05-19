# ruvLLM / esp32-flash / src / ruvector

On-device RuVector layer for the flashable firmware. Smaller surface than `../../../esp32/src/ruvector/` (no `federated_search.rs` / `hyperbolic.rs`).

## Important files
- `mod.rs` - module root.
- `micro_hnsw.rs` - micro-HNSW index for ESP32.
- `rag.rs` - retrieval-augmented generation pipeline.
- `semantic_memory.rs` - episodic semantic memory.
- `anomaly.rs` - lightweight anomaly detector.

## Related
- Larger sibling: `../../../esp32/src/ruvector/`. Federation pairing: `../federation/`.
