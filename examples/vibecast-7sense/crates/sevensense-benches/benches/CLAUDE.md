# sevensense-benches/benches

Criterion benchmark targets exercising the full 7sense stack.

## Files
- `api_benchmark.rs` - End-to-end REST/GraphQL latency through `sevensense-api`.
- `clustering_benchmark.rs` - HDBSCAN / k-means via `sevensense-analysis`.
- `embedding_benchmark.rs` - Perch 2.0 ONNX inference via `sevensense-embedding`.
- `hnsw_benchmark.rs` - HNSW build/query via `sevensense-vector`.

## Run
```
cargo bench -p sevensense-benches
```

## Related
- Helper utilities: `../src/utils.rs`.
- Mirrored workspace-level benches: `../../../benches/`.
