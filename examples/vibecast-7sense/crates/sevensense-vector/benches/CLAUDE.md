# sevensense-vector/benches

Criterion benchmarks for HNSW build and query.

## Files
- `hnsw_benchmark.rs` - Builds an HNSW index of varying sizes/dimensions and measures insert/search throughput and recall.

## Run
```
cargo bench -p sevensense-vector
```

## Related
- Source: `../src/infrastructure/hnsw_index.rs`.
