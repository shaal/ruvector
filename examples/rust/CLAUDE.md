# examples/rust

Loose Rust example files that exercise individual `ruvector-*` crates. These are not a Cargo package on their own; each `.rs` file has its own `main()` and is intended to be copied into a target crate's `examples/` directory or compiled standalone against the relevant crate.

## Files
- `basic_usage.rs` - Create a `VectorDB`, insert vectors, run nearest-neighbor search, basic configuration. Uses `ruvector_core::{VectorDB, VectorEntry, SearchQuery, DbOptions, Result}`.
- `batch_operations.rs` - High-throughput batch inserts/searches with timings.
- `advanced_features.rs` - Hypergraph structures, learned indexes, neural hashing, topological analysis (`ruvector_core::advanced::*`).
- `agenticdb_demo.rs` - Walks the five AgenticDB tables (Reflexion Episodes, Skill Library, Causal Memory, Learning Sessions, Vector DB) via `ruvector_core::AgenticDB`.
- `gnn_example.rs` - Build a `RuvectorLayer` (GNN with MultiHeadAttention + GRU + LayerNorm) from `ruvector_gnn`.
- `rag_pipeline.rs` - Complete RAG pipeline scaffold with MOCK embeddings; documents how to swap in real embedders (`sentence-transformers`, `candle`, ONNX, OpenAI/Anthropic).

## Run
Each file is standalone. Easiest way:
```
cd /home/user/ruvector
cargo run --example basic_usage -p ruvector-core   # only if registered there
# or copy the file into ruvector-core/examples/ then `cargo run --example <name>`
```

## Tech stack
- Rust 2021, `ruvector-core`, `ruvector-gnn`, `rand`.

## Related
- More elaborate Rust examples: the per-topic crates such as `examples/boundary-discovery/`, `examples/google-cloud/`, `examples/vibecast-7sense/`.
