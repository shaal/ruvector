# ruvLLM / esp32 / src / ruvector

On-device RuVector layer: micro-HNSW vector store, RAG, semantic memory, anomaly detection, federated search, and hyperbolic geometry helpers. Built for `no_std` constraints (`heapless`, `fixed`).

## Important files
- `mod.rs` - module root.
- `micro_hnsw.rs` - tiny HNSW index sized for ESP32 SRAM/PSRAM.
- `rag.rs` - retrieval-augmented generation pipeline on top of the micro store.
- `semantic_memory.rs` - episodic semantic memory used by the on-device agent.
- `anomaly.rs` - lightweight anomaly detector over stored vectors.
- `federated_search.rs` - cross-chip search coordination (pairs with `../federation/`).
- `hyperbolic.rs` - hyperbolic-space helpers for hierarchical embeddings.

## Related
- Examples: `../../examples/rag_smart_home.rs`, `space_probe_rag.rs`, `anomaly_industrial.rs`, `swarm_memory.rs`. Host-side counterpart: `../../../src/memory.rs`, `../../../src/router.rs`. Flashable variant: `../../../esp32-flash/src/ruvector/`.
