# ruvector-postgres/src/embeddings

Local embedding generation via fastembed-rs (ONNX-based models). No external API calls. Lazy model loading + thread-safe model cache + batch embedding. Feature-gated under `embeddings`.

Supported models include MiniLM, BGE, MPNet.

## Files

- `mod.rs` — Module entry with SQL-function documentation.
- `models.rs` — Model registry and enum.
- `cache.rs` — Thread-safe model cache (lazy load on first use).
- `functions.rs` — pgrx `#[pg_extern]` SQL functions for text -> embedding.

## Pointers

- Models downloaded via `../../scripts/download_models.rs`.
