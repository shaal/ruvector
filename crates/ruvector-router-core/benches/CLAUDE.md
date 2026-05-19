# ruvector-router-core/benches

Criterion benches for the router-core HNSW + distance pipeline.

## Files

- `vector_search.rs` — `harness = false` bench (declared in `Cargo.toml`) timing HNSW search throughput and distance kernels at
  representative dimensions and dataset sizes.

Run: `cargo bench -p ruvector-router-core`.
