# vibecast-7sense/benches

Workspace-level Criterion benchmarks driven from a separate path (see also `crates/sevensense-benches/`).

## Files
- `api_benchmark.rs` - End-to-end REST/GraphQL latency benchmarks (`sevensense-api`).
- `clustering_benchmark.rs` - HDBSCAN / k-means clustering throughput (`sevensense-analysis`).
- `embedding_benchmark.rs` - Perch 2.0 ONNX embedding generation (`sevensense-embedding`).
- `hnsw_benchmark.rs` - HNSW build and query benchmarks (`sevensense-vector`).
- `utils.rs` - Shared helpers (synthetic data generation, timers).

## Run
```
cargo bench -p sevensense-benches
```

## Related
- Crates being benchmarked are in `../crates/`.
- Aggregating script: `../scripts/run_benchmarks.sh`.
