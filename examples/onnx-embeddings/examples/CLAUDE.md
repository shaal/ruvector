# onnx-embeddings/examples

Cargo `[[example]]` binaries for `ruvector-onnx-embeddings`.

## Files

- `basic.rs` - Smallest possible embed-a-string example.
- `batch.rs` - Batched embedding of multiple texts.
- `semantic_search.rs` - Build a small in-memory index and query semantically.

## How to run

```bash
cargo run --release --example basic_embedding
cargo run --release --example batch_embedding
cargo run --release --example semantic_search
```

## Related

- Library: `../src/embedder.rs`, `../src/model.rs`, `../src/pooling.rs`.
- RuVector integration: `../src/ruvector_integration.rs`.
